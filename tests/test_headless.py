# import settings
# import sys
# import subprocess
# import os
#
# def test_headless():
#     """
#     Maybe this would be better in a shell script than in python
#     """
#     print('launching science birds')
#     cmd = ''
#     if sys.platform == 'darwin':
#         cmd = 'open {}/ab.app'.format(settings.SCIENCE_BIRDS_BIN_DIR)
#     else:
#         cmd = '{}/ScienceBirds_Linux/science_birds_linux.x86_64 {}'. \
#             format(settings.SCIENCE_BIRDS_BIN_DIR,
#                    '-batchmode -nographics' if settings.HEADLESS else '')
#     # Not sure if run will work this way on ubuntu...
#     SB_process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                                        stderr=subprocess.STDOUT,
#                                        shell=True,
#                                        preexec_fn=os.setsid)
#     print('launching java interface')
#     # Popen is necessary as we have to run it in the background
#     cmd2 = '{}{}'.format('xvfb-run ' if settings.HEADLESS else '',
#                          settings.SCIENCE_BIRDS_SERVER_CMD)
#     SB_server_process = subprocess.Popen(cmd2,
#                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
#                                               preexec_fn=os.setsid)
# #    print(SB_server_process.communicate()[0])
#     print('done')
#     # sb_client.ar.configure(self.sb_client.id)
#     # levels = self.sb_client.update_no_of_levels()
#     #
#     # self.sb_client.solved = [0 for x in range(levels)]
#     # self.sb_client.current_level = self.sb_client.get_next_level()
#     # self.sb_client.ar.load_level(self.sb_client.current_level)
#     # print('solving level: {}'.format(self.sb_client.current_level))


