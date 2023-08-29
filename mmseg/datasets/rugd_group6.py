from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
# class RUGDDataset_Group6(CustomDataset):
#     """RUGD dataset.
#
#     """
#
#
#
#     CLASSES = ("background", "T", "R")
#
#     PALETTE = [[ 0, 0, 0 ], [ 0,255,0 ],[ 0, 0, 255 ] ]
#
#
#     def __init__(self, **kwargs):
#         super(RUGDDataset_Group6, self).__init__(
#             img_suffix='.png',
#             seg_map_suffix='_group6.png',
#             # seg_map_suffix='_labelid_novoid_255.png',
#             **kwargs)
#         self.CLASSES = ("background", "T", "R")
#         self.PALETTE = [[ 0, 0, 0 ], [ 0,255,0 ],[ 0, 0, 255 ] ]
#         # assert osp.exists(self.img_dir)
class RUGDDataset_Group6(CustomDataset):
    """RUGD dataset.

    """



    CLASSES = ( "background", "apple", 'ball',
        'banana',
        'bell_pepper',
        'binder',
        'bowl',
        'cereal_box',
        'coffee_mug',
        'flashlight',
        'food_bag',
        'food_box',
        'food_can',
        'glue_stick',
        'hand_towel',
        'instant_noodles',
        'keyboard',
        'kleenex',
        'lemon',
        'lime',
        'marker',
        'orange',
        'peach',
        'pear',
        'potato',
        'shampoo',
        'soda_can',
        'sponge',
        'stapler',
        'tomato',
        'toothpaste',
        'unknown')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],]


    def __init__(self, **kwargs):
        super(RUGDDataset_Group6, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_group6.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "apple", 'ball',
        'banana',
        'bell_pepper',
        'binder',
        'bowl',
        'cereal_box',
        'coffee_mug',
        'flashlight',
        'food_bag',
        'food_box',
        'food_can',
        'glue_stick',
        'hand_towel',
        'instant_noodles',
        'keyboard',
        'kleenex',
        'lemon',
        'lime',
        'marker',
        'orange',
        'peach',
        'pear',
        'potato',
        'shampoo',
        'soda_can',
        'sponge',
        'stapler',
        'tomato',
        'toothpaste',
        'unknown')
        self.PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],]

