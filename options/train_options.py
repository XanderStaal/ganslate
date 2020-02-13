from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--update_html_freq', type=int, default=50, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=25, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr_G', type=float, default=0.0002, help='initial generator\'s learning rate for adam')
        parser.add_argument('--lr_D', type=float, default=0.0002, help='initial discriminator\'s learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--grad_reg', type=float, default=0.0, help='Use consensus optimization (test)')
        parser.add_argument('--D_rollout', type=int, default=1, help='Discriminator rollout (test)')

        parser.add_argument('--patch_size', required=True, nargs='+', type=int, help='Size of patches extracted from volumes (z,x,y). Input example: 64 64 64')
        parser.add_argument('--threshold_black_voxels', type=float, default=1.0, help='Threshold ratio: number of completely black voxels / total number of voxels. Defines how many completely black voxels are allowed. Default 1, no threshold is happening.')
        parser.add_argument("--wandb", help="Use Weights&Biases (wandb.com) to track the experiment", default=False, type=bool)
        self.isTrain = True
        return parser
