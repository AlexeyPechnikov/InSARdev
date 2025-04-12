# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_prm import S1_prm
from .PRM import PRM

class S1_gmtsar(S1_prm):

    def _ext_orb_s1a(self, burst, debug=False):
        """
        Extracts orbital data for the Sentinel-1A satellite by running GMTSAR binary `ext_orb_s1a`.

        Parameters
        ----------
        stem : str
            Stem name used for file naming.
        date : str, optional
            Date for which to extract the orbital data. If not provided or if date is the reference, 
            it will extract the orbital data for the reference. Defaults to None.
        debug : bool, optional
            If True, prints debug information. Defaults to False.

        Examples
        --------
        _ext_orb_s1a(1, 'stem_name', '2023-05-24', True)
        """
        import os
        import subprocess

        df = self.get_record(burst)
        prefix = self.get_prefix(burst)
        basedir = os.path.join(self.basedir, prefix)

        orbit = df['orbit'].iloc[0]
        orbitfile = os.path.join(self.datadir, orbit)
        orbitfile = os.path.relpath(orbitfile, basedir)

        argv = ['ext_orb_s1a', f'{burst}.PRM', orbitfile, burst]
        if debug:
            print ('DEBUG: argv', argv)
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', cwd=basedir)
        stdout_data, stderr_data = p.communicate()
        if len(stderr_data) > 0 and debug:
            print ('DEBUG: ext_orb_s1a', stderr_data)
        if len(stdout_data) > 0 and debug:
            print ('DEBUG: ext_orb_s1a', stdout_data)

        return

    # produce LED and PRM in basedir
    # when date=None work on reference scene
    def _make_s1a_tops(self, burst, mode=0, rshift_fromfile=None, ashift_fromfile=None, debug=False):
        """
        Produces LED and PRM in the base directory by executing GMTSAR binary `make_s1a_tops`.

        Parameters
        ----------
        date : str, optional
            Date for which to create the Sentinel-1A TOPS products. If not provided, 
            it processes the reference image. Defaults to None.
        mode : int, optional
            Mode for `make_s1a_tops` script: 
            0 - no SLC; 
            1 - center SLC; 
            2 - high SLCH and low SLCL; 
            3 - output ramp phase.
            Defaults to 0.
        rshift_fromfile : str, optional
            Path to the file with range shift data. Defaults to None.
        ashift_fromfile : str, optional
            Path to the file with azimuth shift data. Defaults to None.
        debug : bool, optional
            If True, prints debug information. Defaults to False.

        Notes
        -----
        The function executes an external binary `make_s1a_tops`.
        Also, this function calls the `ext_orb_s1a` method internally.

        Examples
        --------
        _make_s1a_tops(1, '2023-05-24', 1, '/path/to/rshift.grd', '/path/to/ashift.grd', True)
        """
        import os
        import subprocess

        df = self.get_record(burst)
        prefix = self.get_prefix(burst)
        basedir = os.path.join(self.basedir, prefix)

        # if os.path.dirname(prefix_path) == '':
        #     basedir = self.basedir
        # else:
        #     basedir = os.path.join(self.basedir, os.path.dirname(prefix))
        #     prefix = os.path.basename(prefix)
        #     if rshift_fromfile is not None:
        #         rshift_fromfile = os.path.basename(rshift_fromfile)
        #     if ashift_fromfile is not None:
        #         ashift_fromfile = os.path.basename(ashift_fromfile)

        #basedir = os.path.join(self.basedir, os.path.dirname(prefix))
        #print ('basedir', basedir)
        xmlfile = os.path.join(self.datadir, prefix, 'annotation', f'{burst}.xml')
        xmlfile = os.path.relpath(xmlfile, basedir)
        tiffile = os.path.join(self.datadir, prefix, 'measurement', f'{burst}.tiff')
        tiffile = os.path.relpath(tiffile, basedir)

        #argv = ['make_s1a_tops', xmlfile, tiffile, f'{prefix}/{burst}', str(mode)]
        argv = ['make_s1a_tops', xmlfile, tiffile, burst, str(mode)]
        if rshift_fromfile is not None:
            argv.append(rshift_fromfile)
        if ashift_fromfile is not None:
            argv.append(ashift_fromfile)
        if debug:
            print ('DEBUG: argv', argv)
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', cwd=basedir)
        stdout_data, stderr_data = p.communicate()
        if len(stderr_data) > 0 and debug:
            print ('DEBUG: make_s1a_tops', stderr_data)
        if len(stdout_data) > 0 and debug:
            print ('DEBUG: make_s1a_tops', stdout_data)

        self._ext_orb_s1a(burst, debug=debug)

        return

