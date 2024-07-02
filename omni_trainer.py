
import os
import sys
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from utils import DiceLoss
from datasets.dataset import USdatasetCls, USdatasetSeg
from datasets.omni_dataset import WeightedRandomSamplerDDP
from datasets.omni_dataset import USdatasetOmni_cls, USdatasetOmni_seg
from datasets.dataset import RandomGenerator, CenterCropGenerator
from sklearn.metrics import roc_auc_score
from utils import omni_seg_test


def omni_train(args, model, snapshot_path):

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=7200))

    if int(os.environ["LOCAL_RANK"]) == 0:
        print('** GPU NUM ** : ', torch.cuda.device_count())
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    print(f"** DDP ** : Start running on rank {rank}.")

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train_seg = USdatasetOmni_seg(base_dir=args.root_path, split="train", transform=transforms.Compose(
        [RandomGenerator(output_size=[args.img_size, args.img_size])]), prompt=args.prompt)

    weight_base = [1/4, 1/2, 2, 2, 1, 2, 2]
    sample_weight_seq = [[weight_base[dataset_index]] *
                         element for dataset_index, element in enumerate(db_train_seg.subset_len)]
    sample_weight_seq = [element for sublist in sample_weight_seq for element in sublist]

    weighted_sampler_seg = WeightedRandomSamplerDDP(
        data_set=db_train_seg,
        weights=sample_weight_seq,
        num_replicas=world_size,
        rank=rank,
        num_samples=args.num_samples_seg,
        replacement=True
    )
    trainloader_seg = DataLoader(db_train_seg,
                                 batch_size=batch_size,
                                 num_workers=16,
                                 pin_memory=True,
                                 worker_init_fn=worker_init_fn,
                                 sampler=weighted_sampler_seg
                                 )

    db_train_cls = USdatasetOmni_cls(base_dir=args.root_path, split="train", transform=transforms.Compose(
        [RandomGenerator(output_size=[args.img_size, args.img_size])]), prompt=args.prompt)

    weight_base = [2, 1/4, 2, 2]
    sample_weight_seq = [[weight_base[dataset_index]] *
                         element for dataset_index, element in enumerate(db_train_cls.subset_len)]
    sample_weight_seq = [element for sublist in sample_weight_seq for element in sublist]

    weighted_sampler_cls = WeightedRandomSamplerDDP(
        data_set=db_train_cls,
        weights=sample_weight_seq,
        num_replicas=world_size,
        rank=rank,
        num_samples=args.num_samples_cls,
        replacement=True
    )
    trainloader_cls = DataLoader(db_train_cls,
                                 batch_size=batch_size,
                                 num_workers=16,
                                 pin_memory=True,
                                 worker_init_fn=worker_init_fn,
                                 sampler=weighted_sampler_cls
                                 )

    model = model.to(device=device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)

    model.train()

    seg_ce_loss = CrossEntropyLoss()
    seg_dice_loss = DiceLoss()
    cls_ce_loss = CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05, betas=(0.9, 0.999))

    resume_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location='cpu')['model'])
        optimizer.load_state_dict(torch.load(args.resume, map_location='cpu')['optimizer'])
        resume_epoch = torch.load(args.resume, map_location='cpu')['epoch']

    writer = SummaryWriter(snapshot_path + '/log')
    global_iter_num = 0
    seg_iter_num = 0
    cls_iter_num = 0
    max_epoch = args.max_epochs
    total_iterations = (len(trainloader_seg) + len(trainloader_cls))
    max_iterations = args.max_epochs * total_iterations
    logging.info("{} batch size. {} iterations per epoch. {} max iterations ".format(
        batch_size, total_iterations, max_iterations))
    best_performance = 0.0
    best_epoch = 0

    if int(os.environ["LOCAL_RANK"]) != 0:
        iterator = tqdm(range(resume_epoch, max_epoch), ncols=70, disable=True)
    else:
        iterator = tqdm(range(resume_epoch, max_epoch), ncols=70, disable=False)

    for epoch_num in iterator:
        logging.info("\n epoch: {}".format(epoch_num))
        weighted_sampler_seg.set_epoch(epoch_num)
        weighted_sampler_cls.set_epoch(epoch_num)

        torch.cuda.empty_cache()
        for i_batch, sampled_batch in tqdm(enumerate(trainloader_seg)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([
                    1, 0]).float().to(device=device)
                task_prompt = torch.tensor(np.array(sampled_batch['task_prompt'])).permute([
                    1, 0]).float().to(device=device)
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([
                    1, 0]).float().to(device=device)
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([
                    1, 0]).float().to(device=device)
                (x_seg, _) = model((image_batch, position_prompt, task_prompt, type_prompt, nature_prompt))
            else:
                (x_seg, _) = model(image_batch)

            loss_ce = seg_ce_loss(x_seg, label_batch[:].long())
            loss_dice = seg_dice_loss(x_seg, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - global_iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            seg_iter_num = seg_iter_num + 1
            global_iter_num = global_iter_num + 1

            writer.add_scalar('info/lr', lr_, seg_iter_num)
            writer.add_scalar('info/seg_loss', loss, seg_iter_num)

            logging.info('global iteration %d and seg iteration %d : loss : %f' %
                         (global_iter_num, seg_iter_num, loss.item()))

        torch.cuda.empty_cache()
        for i_batch, sampled_batch in tqdm(enumerate(trainloader_cls)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([
                    1, 0]).float().to(device=device)
                task_prompt = torch.tensor(np.array(sampled_batch['task_prompt'])).permute([
                    1, 0]).float().to(device=device)
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([
                    1, 0]).float().to(device=device)
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([
                    1, 0]).float().to(device=device)
                (_, x_cls) = model((image_batch, position_prompt, task_prompt, type_prompt, nature_prompt))
            else:
                (_, x_cls) = model(image_batch)

            loss_ce = cls_ce_loss(x_cls, label_batch[:].long())
            loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - global_iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            cls_iter_num = cls_iter_num + 1
            global_iter_num = global_iter_num + 1

            writer.add_scalar('info/lr', lr_, cls_iter_num)
            writer.add_scalar('info/cls_loss', loss, cls_iter_num)

            logging.info('global iteration %d and cls iteration %d : loss : %f' %
                         (global_iter_num, cls_iter_num, loss.item()))

        dist.barrier()

        if int(os.environ["LOCAL_RANK"]) == 0:
            torch.cuda.empty_cache()

            save_dict = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch_num}
            save_latest_path = os.path.join(snapshot_path, 'latest_{}.pth'.format(epoch_num))
            if os.path.exists(os.path.join(snapshot_path, 'latest_{}.pth'.format(epoch_num-1))):
                os.remove(os.path.join(snapshot_path, 'latest_{}.pth'.format(epoch_num-1)))
                os.remove(os.path.join(snapshot_path, 'latest.pth'))
            torch.save(save_dict, save_latest_path)
            os.system('ln -s ' + os.path.abspath(save_latest_path) + ' ' + os.path.join(snapshot_path, 'latest.pth'))

            model.eval()
            total_performance = 0.0

            seg_val_set = ["BUS-BRA", "BUSIS", "CAMUS", "DDTI", "Fetal_HC", "kidneyUS", "UDIAT"]
            seg_avg_performance = 0.0

            for dataset_name in seg_val_set:
                num_classes = 2
                db_val = USdatasetSeg(
                    base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
                    split="val",
                    list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
                    transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                    prompt=args.prompt
                )
                val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)
                logging.info("{} val iterations per epoch".format(len(val_loader)))

                metric_list = 0.0
                count_matrix = np.ones((len(db_val), num_classes-1))
                for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    if args.prompt:
                        position_prompt = torch.tensor(
                            np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                        task_prompt = torch.tensor(
                            np.array([[1]*position_prompt.shape[0], [0]*position_prompt.shape[0]])).permute([1, 0]).float()
                        type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                        nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                        metric_i = omni_seg_test(image, label, model,
                                                 classes=num_classes,
                                                 prompt=args.prompt,
                                                 type_prompt=type_prompt,
                                                 nature_prompt=nature_prompt,
                                                 position_prompt=position_prompt,
                                                 task_prompt=task_prompt
                                                 )
                    else:
                        metric_i = omni_seg_test(image, label, model,
                                                 classes=num_classes)

                    for sample_index in range(len(metric_i)):
                        if not metric_i[sample_index][1]:
                            count_matrix[i_batch*batch_size+sample_index, 0] = 0
                    metric_i = [element[0] for element in metric_i]
                    metric_list += np.array(metric_i).sum()

                metric_list = metric_list / (count_matrix.sum(axis=0) + 1e-6)
                performance = np.mean(metric_list, axis=0)

                writer.add_scalar('info/val_seg_metric_{}'.format(dataset_name), performance, epoch_num)

                seg_avg_performance += performance

            seg_avg_performance = seg_avg_performance / len(seg_val_set)
            total_performance += seg_avg_performance
            writer.add_scalar('info/val_metric_seg_Total', seg_avg_performance, epoch_num)

            cls_val_set = ["Appendix", "BUS-BRA", "Fatty-Liver", "UDIAT"]
            cls_avg_performance = 0.0

            for dataset_name in cls_val_set:
                num_classes = 2
                db_val = USdatasetCls(
                    base_dir=os.path.join(args.root_path, "classification", dataset_name),
                    split="val",
                    list_dir=os.path.join(args.root_path, "classification", dataset_name),
                    transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                    prompt=args.prompt
                )

                val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)
                logging.info("{} val iterations per epoch".format(len(val_loader)))
                model.eval()

                label_list = []
                prediction_prob_list = []
                for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    if args.prompt:
                        position_prompt = torch.tensor(
                            np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                        task_prompt = torch.tensor(
                            np.array([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]])).permute([1, 0]).float()
                        type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                        nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                        with torch.no_grad():
                            output = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(),
                                           type_prompt.cuda(), nature_prompt.cuda()))[1]
                    else:
                        with torch.no_grad():
                            output = model(image.cuda())[1]
                    output_prob = torch.softmax(output, dim=1).data.cpu().numpy()

                    label_list.append(label.numpy())
                    prediction_prob_list.append(output_prob)

                label_list = np.expand_dims(np.concatenate(
                    (np.array(label_list[:-1]).flatten(), np.array(label_list[-1]).flatten())), axis=1).astype('uint8')
                label_list_OneHot = np.eye(num_classes)[label_list].squeeze(1)
                performance = roc_auc_score(label_list_OneHot, np.concatenate(
                    (np.array(prediction_prob_list[:-1]).reshape(-1, 2), prediction_prob_list[-1])), multi_class='ovo')

                writer.add_scalar('info/val_cls_metric_{}'.format(dataset_name), performance, epoch_num)

                cls_avg_performance += performance

            cls_avg_performance = cls_avg_performance / len(cls_val_set)
            total_performance += cls_avg_performance
            writer.add_scalar('info/val_metric_cls_Total', cls_avg_performance, epoch_num)

            TotalAvgPerformance = total_performance/2

            logging.info('This epoch %d Validation performance: %f' % (epoch_num, TotalAvgPerformance))
            logging.info('But the best epoch is: %d and performance: %f' % (best_epoch, best_performance))
            writer.add_scalar('info/val_metric_TotalMean', TotalAvgPerformance, epoch_num)
            if TotalAvgPerformance >= best_performance:
                if os.path.exists(os.path.join(snapshot_path, 'best_model_{}_{}.pth'.format(best_epoch, round(best_performance, 4)))):
                    os.remove(os.path.join(snapshot_path, 'best_model_{}_{}.pth'.format(
                        best_epoch, round(best_performance, 4))))
                    os.remove(os.path.join(snapshot_path, 'best_model.pth'))
                best_epoch = epoch_num
                best_performance = TotalAvgPerformance
                logging.info('Validation TotalAvgPerformance in best val model: %f' % (TotalAvgPerformance))
                save_model_path = os.path.join(snapshot_path, 'best_model_{}_{}.pth'.format(
                    epoch_num, round(best_performance, 4)))
                os.system('ln -s ' + os.path.abspath(save_model_path) +
                          ' ' + os.path.join(snapshot_path, 'best_model.pth'))
                torch.save(model.state_dict(), save_model_path)
                logging.info("save model to {}".format(save_model_path))

        model.train()

    writer.close()
    return "Training Finished!"
