from data_provider.data_loader import Dataset_CSI300
from torch.utils.data import DataLoader

data_dict = {
    'CSI300': Dataset_CSI300
}


def data_provider(args, flag):
    """
    flag: 'train' | 'val' | 'test'
    """
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        target=args.target
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    print(f"{flag} set: {len(data_set)} samples")
    return data_set, data_loader