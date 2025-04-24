# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_slc import S1_slc
from .PRM import PRM

class S1_prm(S1_slc):

    def PRM(self, burst: str, basedir) -> PRM:
        """
        Open a PRM (Parameter) file.

        Parameters
        ----------
        date : str, optional
            The date of the PRM file. If None or equal to self.reference, return the reference PRM file. Default is None.
        multi : bool, optional
            If True, open a multistem PRM file. If False, open a stem PRM file. Default is True.
        
        Returns
        -------
        PRM
            An instance of the PRM class representing the opened PRM file.
        """
        import os

        #prefix = self.get_prefix(burst)
        #print ('PRM prefix', prefix)
        filename = os.path.join(basedir, f'{burst}.PRM')
        #print ('PRM filename', filename)
        return PRM.from_file(filename)


