# from dataloader import dataset
# from torch.utils.data import DataLoader

# def make_dataloader(args, **kwargs):
#     print('dataset: ', args.dataset)

#     if args.dataset == 'REFUGE':   
#         args.data_path = '/data2/dingfei/datasets/REFUGE/GON/segmentation'
#         args.base_size = 513
#         args.crop_size = 513
#         args.input_size = 513
#         train_set = getattr(dataset, args.dataset)(args, ['train'])
#         test_set = getattr(dataset, args.dataset)(args, ['test'])
# #         val_set = None
#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = None
#         test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
#     elif args.dataset == 'TR9874':   
#         args.data_path = '/data/dingfei/datasets/eye/10053'
#         args.base_size = 256
#         args.crop_size = 256
#         args.input_size = 256
#         train_set = getattr(dataset, args.dataset)(args, ['train'])
#         val_set = getattr(dataset, args.dataset)(args, ['val'])
#         test_set = getattr(dataset, args.dataset)(args, ['test'])
#         num_class = test_set.NUM_CLASSES
        
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
#         test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
#     elif args.dataset == 'DGS':   
#         args.data_path = '../datasets/DGS'
#         args.base_size = 513
#         args.crop_size = 513
#         args.input_size = 513
#         #train_set = getattr(dataset, args.dataset)(args, ['train'])
#         test_set = getattr(dataset, args.dataset)(args, ['train','test'])
# #         val_set = None
#         num_class = test_set.NUM_CLASSES
        
#         train_loader = None
#         val_loader = None
#         test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

#     elif args.dataset == 'LUNA':   
#         args.data_path = '../datasets/LUNA'
#         args.base_size = 513
#         args.crop_size = 513
#         args.input_size = 513
#         #train_set = getattr(dataset, args.dataset)(args, ['train'])
#         test_set = getattr(dataset, args.dataset)(args, ['test'])
# #         val_set = None
#         num_class = train_set.NUM_CLASSES
#         train_loader = None
#         val_loader = None
#         test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        
#     elif args.dataset == 'DRIVE':   
#         args.data_path = '../datasets/DRIVE'
#         args.input_size = 224
#         #train_set = None
#         #val_set = None
#         test_set = getattr(dataset, args.dataset)(args, ['test'])
#         num_class = test_set.NUM_CLASSES
#         train_loader = None
#         val_loader = None
#         test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        
#     return train_loader, val_loader, test_loader, num_class