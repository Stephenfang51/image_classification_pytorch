def inference(model):
    test_transform = transforms.Compose([transforms.Resize((int(Size), int(Size))),
                                         # transforms.TenCrop(Size),
                                         # Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    dst_test = CDCDataset(test_data, transform=test_transform, mode='test')
    dataloader_test = DataLoader(dst_test, shuffle=False, batch_size=Batch_size // 2, num_workers=8)

    model.eval()
    results = []
    print('inferencing..')
    for ims, im_names in dataloader_test:
        input = ims.requires_grad_().cuda()
        output = model(input)
        _, preds = output.topk(1, 1, True, True)  # 取top 1
        preds = preds.cpu().detach().numpy()
        for pred, im_name in zip(preds, im_names):
            top1_name = [list(cls2label.keys())  # 字典取key
                         [list(cls2label.values()).index(p)]  # 字典取value
                         for p in pred]
            # cls2label = {'cgm': 0, 'cmd': 1, 'healthy': 2, 'cbb': 3, 'cbsd': 4}
            results.append({'Id': im_name, 'Category': ''.join(top1_name)})
    df = pd.DataFrame(results, columns=['Category', 'Id'])
    df.to_csv('sub.csv', index=False)