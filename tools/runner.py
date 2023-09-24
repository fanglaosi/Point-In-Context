from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
from datasets import build_dataset_from_cfg
import torch.utils.data
from utils.misc import *

class Loss_Metric:
    def __init__(self, loss = 0.):
        if type(loss).__name__ == 'dict':
            self.loss = loss['loss']
        else:
            self.loss = loss

    def better_than(self, other):
        if self.loss < other.loss:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['loss'] = self.loss
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    train_dataset = build_dataset_from_cfg(config.dataset.train._base_, config.dataset.train.others)

    test_dataset = build_dataset_from_cfg(config.dataset.test._base_, config.dataset.test.others)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.total_bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.total_bs * 2,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   num_workers=int(args.num_workers),
                                                   worker_init_fn=worker_init_fn)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Loss_Metric(100000.)
    metrics = Loss_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Loss_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    print_log('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (pointset1_pc, pointset2_pc, target1, target2) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)

            pointset1_pc = pointset1_pc.cuda()
            pointset2_pc = pointset2_pc.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

            _, _, loss = base_model(pointset1_pc, pointset2_pc, target1, target2)
            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item()*1000])

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    mean_loss = 0
    i = 0
    with torch.no_grad():
        for idx, (pointset1_pc, pointset2_pc, target1, target2) in enumerate(test_dataloader):
            pointset1_pc = pointset1_pc.cuda()
            pointset2_pc = pointset2_pc.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

            _, _, loss = base_model(pointset1_pc, pointset2_pc, target1, target2)
            loss = loss.mean()
            mean_loss += loss.item()*1000
            i += 1
        mean_loss /= i
        print_log('[Validation] EPOCH: %d  loss = %.4f' % (epoch, mean_loss), logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/Loss', mean_loss, epoch)

    return Loss_Metric(mean_loss)


def test_net():
    pass