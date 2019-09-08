import logging
from tensorboard import program
from tensorboard import default
import sys

class TensorboardHelper:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def run(self):
        # Remove http messages
        log = logging.getLogger('tensorflow').setLevel(logging.FATAL)
        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', self.dir_path])
        url = tb.launch()
        sys.stdout.write('TensorBoard at %s \n' % url)

# Starting tensorboard server
tensorboardServer = TensorboardHelper(".")
tensorboardServer.run()

input("You can now go to Tensorboard url and look into the statistics (if training was performed).\nTo end session and kill Tensorboard instance input any key...")