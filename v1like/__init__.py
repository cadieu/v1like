import v1like_math
import v1like_funcs
import config

import os
from v1like_extract import v1like_fromarray, MinMaxError
from scipy.misc import fromimage
from PIL import Image

# Let's make a python interface for this:
class V1Like(object):

    def __init__(self,config):
        """ Class for computing V1like output on an image

        Inputs:
          config -- string indicating V1 configuration ('v1like_a', 'v1like_a_plus', etc.)

        """

        assert isinstance(config,(str,unicode))

        self.config = config

        config_fname = os.path.join(os.path.split(__file__)[0],'config',self.config.lower() + '.py')

        # copying from v1like_extract.v1like_fromfilename
        self.v1like_config = {}
        execfile(config_fname, {}, self.v1like_config)

        self.model = self.v1like_config['model']
        if len(self.model) != 1:
            raise NotImplementedError

        self.rep, self.featsel = self.model[0]

        resize_type = self.rep['preproc'].get('resize_type', 'input')
        if resize_type == 'output':
            if 'max_edge' not in self.rep['preproc']:
                raise NotImplementedError
            # add whatever is needed to get output = max_edge
            new_max_edge = self.rep['preproc']['max_edge']

            preproc_lsum = self.rep['preproc']['lsum_ksize']
            new_max_edge += preproc_lsum-1

            normin_kshape = self.rep['normin']['kshape']
            assert normin_kshape[0] == normin_kshape[1]
            new_max_edge += normin_kshape[0]-1

            filter_kshape = self.rep['filter']['kshape']
            assert filter_kshape[0] == filter_kshape[1]
            new_max_edge += filter_kshape[0]-1

            normout_kshape = self.rep['normout']['kshape']
            assert normout_kshape[0] == normout_kshape[1]
            new_max_edge += normout_kshape[0]-1

            pool_lsum = self.rep['pool']['lsum_ksize']
            new_max_edge += pool_lsum-1

            self.rep['preproc']['max_edge'] = new_max_edge

    def __call__(self, imgarr):
        """ Compute V1like output on an image

        Inputs:
          imgarr -- 2d numpy array

        Outputs:
          fvector -- feature vector output
        """

        img = Image.fromarray(imgarr)

        # do resizing
        if 'max_edge' in self.rep['preproc']:
            max_edge = self.rep['preproc']['max_edge']
            resize_method = self.rep['preproc']['resize_method']

            if max_edge is not None:
                # -- resize so that the biggest edge is max_edge (keep aspect ratio)
                iw, ih = img.size
                if iw > ih:
                    new_iw = max_edge
                    new_ih = int(round(1.* max_edge * ih/iw))
                else:
                    new_iw = int(round(1.* max_edge * iw/ih))
                    new_ih = max_edge
                if resize_method.lower() == 'bicubic':
                    img = img.resize((new_iw, new_ih), Image.BICUBIC)
                elif resize_method.lower() == 'antialias':
                    img = img.resize((new_iw, new_ih), Image.ANTIALIAS)
                else:
                    raise ValueError("resize_method '%s' not understood", resize_method)

            # -- convert to a numpy array
            imgarr = fromimage(img)#/255.

        else:
            resize = self.rep['preproc']['resize']
            resize_method = self.rep['preproc']['resize_method']

            # -- resize image if needed
            if resize is not None:
                rtype, rsize = resize

                if rtype == 'min_edge':
                    # -- resize so that the smallest edge is rsize (keep aspect ratio)
                    iw, ih = img.size
                    if iw < ih:
                        new_iw = rsize
                        new_ih = int(round(1.* rsize * ih/iw))
                    else:
                        new_iw = int(round(1.* rsize * iw/ih))
                        new_ih = rsize

                elif rtype == 'max_edge':
                    # -- resize so that the biggest edge is rszie (keep aspect ratio)
                    iw, ih = img.size
                    if iw > ih:
                        new_iw = rsize
                        new_ih = int(round(1.* rsize * ih/iw))
                    else:
                        new_iw = int(round(1.* rsize * iw/ih))
                        new_ih = rsize

                else:
                    raise ValueError, "resize parameter not understood"

                if resize_method.lower() == 'bicubic':
                    img = img.resize((new_iw, new_ih), Image.BICUBIC)
                elif resize_method.lower() == 'antialias':
                    img = img.resize((new_iw, new_ih), Image.ANTIALIAS)
                else:
                    raise ValueError("resize_method '%s' not understood", resize_method)

            # -- convert to a numpy array
            imgarr = fromimage(img)#/255.

        try:
            fvector = v1like_fromarray(imgarr, self.rep, self.featsel)
        except MinMaxError, err:
            raise MinMaxError
        except AssertionError, err:
            raise err

        return fvector