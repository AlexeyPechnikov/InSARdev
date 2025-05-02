# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
import io
import os
import requests
import zipfile
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

class HTTPRangeReader(io.RawIOBase):
    """
    A RawIOBase wrapper that does HTTP Range requests on demand,
    reusing a single Session + HTTPAdapter for keep-alive and retries,
    and feeding a tqdm bar as it reads.
    """
    def __init__(self, url: str, max_retries=3, backoff=0.5):
        self.url = url
        # fetch total size once
        head = requests.head(url, allow_redirects=True)
        head.raise_for_status()
        self.total_size = int(head.headers.get("Content-Length", 0))
        self.pos = 0

        # set up session + retries
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # progress bar for actual bytes read
        self.bar = tqdm(
            total=self.total_size,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=os.path.basename(url),
        )

    def readinto(self, b: bytearray) -> int:
        if self.pos >= self.total_size:
            return 0
        length = len(b)
        end = min(self.pos + length - 1, self.total_size - 1)
        headers = {"Range": f"bytes={self.pos}-{end}"}
        r = self.session.get(self.url, headers=headers, stream=True)
        r.raise_for_status()
        # read directly into buffer
        n = r.raw.readinto(b)
        self.pos += n
        self.bar.update(n)
        return n

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total_size + offset
        else:
            raise ValueError("Invalid whence")
        self.pos = max(0, min(self.pos, self.total_size))
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        self.bar.close()
        return super().close()

def unzip(url: str, target_dir: str, buffer_size: int = 4 << 20):
    """
    Stream-download unpacking remote ZIP content via HTTP Range requests.
    Skips already-extracted files (filling the bar by the compressed size).
    Works well with large files on Zenodo not hummering the server.
    """
    target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    raw = HTTPRangeReader(url)
    buf = io.BufferedReader(raw, buffer_size=buffer_size)

    with zipfile.ZipFile(buf) as zf:
        for info in zf.infolist():
            # skip macOS metadata folder contents
            # it is empty and useless but requires extra long time to extract
            if info.filename.startswith("__MACOSX/"):
                continue
            
            out_path = os.path.join(target_dir, info.filename)

            # create empty directories listed in zip
            if info.is_dir() or info.filename.endswith("/"):
                os.makedirs(out_path, exist_ok=True)
                continue

            # skip if already extracted and complete
            if os.path.exists(out_path) and os.path.getsize(out_path) == info.file_size:
                # account compressed bytes for progress
                raw.bar.update(info.compress_size)
                continue

            # extract creating non-empty unlisted directories for all files
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=64 << 10)

    # downloaded size does not include zip header, central directory, etc.
    # to make progress bar complete, we need to add the remaining size
    remaining = raw.total_size - raw.bar.n
    if remaining > 0:
        raw.bar.update(remaining)
    raw.close()

def download(url: str, dst: str, chunk_size: int = 2**20):
    import os
    import requests
    from tqdm.auto import tqdm

    downloaded = os.path.getsize(dst) if os.path.exists(dst) else 0
    head = requests.head(url, allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers.get("content-length", 0))
    #print ('downloaded', downloaded, 'expected', expected)
    if downloaded and expected and downloaded >= expected:
        print(f"{dst} is already fully downloaded.")
        return
    if os.path.exists(dst):
        print(f'{dst} is incompletely downloaded, removing it.')
        os.remove(dst)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dst, 'wb') as f, \
         tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc=os.path.basename(dst)) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            bar.update(len(chunk))
