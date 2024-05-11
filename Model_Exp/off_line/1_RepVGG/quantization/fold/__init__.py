from .convbnact import Conv2dBNReLU
from .convbn import Conv2dBN
from .sharequant import ShareQuant

from .sabn import SA2dBN
from .sabnact import SA2dBNReLU

from .fold_op import fold_CBR, fold_the_network, make_the_shortcut_share, fold_CB
from .foldmodule import isfoldmodule