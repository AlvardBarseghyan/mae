model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model testing settings
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# # dataset settings
# dataset_type = 'BDDVideoDataset'
# data_root = 'data/bdd/bdd100k/'
# ann_root = 'data/bdd/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=2,
#     # train=[
#     #     dict(
#     #         type=dataset_type,
#     #         load_as_video=False,
#     #         ann_file=ann_root +
#     #                  'annotations/det_20/det_train_cocofmt.json',
#     #         img_prefix=data_root + 'images/100k/train/',
#     #         pipeline=train_pipeline)
#     # ],
#     val=dict(
#         type=dataset_type,
#         ann_file=ann_root +
#                  'annotations/box_track_20/box_track_val_cocofmt.json',
#         scalabel_gt = ann_root + 'annotations/scalabel_gt/box_track_20/val/',
#         img_prefix=data_root + 'images/track/val/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=ann_root +
#                  'annotations/box_track_20/box_track_val_cocofmt.json',
#         scalabel_gt=ann_root + 'annotations/scalabel_gt/box_track_20/val/',
#         img_prefix=data_root + 'images/track/val/',
#         pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'track'], interval=1)
