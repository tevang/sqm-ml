import sys
from inspect import getframeinfo, stack


class bcolors:
    ## See also https://pypi.python.org/pypi/blessings/   https://pypi.python.org/pypi/colorama
    HEADER = '\033[1m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    BOLDGREEN = '\x1b[32;1m'
    BOLDBLUE = '\x1b[34;1m'
    FAIL = '\x1b[91;1m'
    WARNING = '\033[103m' # white font with yellow background '\033[45m'
    OKRED = '\033[91m'
    ENDC = '\033[0m'
    ENDBOLD = '\x1b[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.BOLDGREEN = ''
        self.OKRED = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''
        self.ENDBOLD = ''


class ColorPrint():

    def __init__(self, message, type):
        prefix = {
            "HEADER": bcolors.HEADER,
            "OKBLUE": bcolors.OKBLUE,
            "OKGREEN": bcolors.OKGREEN,
            "BOLDGREEN": bcolors.BOLDGREEN,
            "BOLDBLUE": bcolors.BOLDBLUE,
            "OKRED": bcolors.OKRED,
            "WARNING": bcolors.WARNING,
            "FAIL": bcolors.FAIL,
        }
        suffix = {
            "HEADER" : bcolors.ENDBOLD,
            "OKBLUE" : bcolors.ENDC,
            "OKGREEN" : bcolors.ENDC,
            "BOLDGREEN" : bcolors.ENDBOLD,
            "BOLDBLUE" : bcolors.ENDBOLD,
            "OKRED" : bcolors.ENDC,
            "WARNING" : bcolors.ENDC,
            "FAIL" : bcolors.ENDBOLD,
        }
        print(prefix[type] + message + suffix[type])


def Debuginfo(message, fail=False):
    caller = getframeinfo(stack()[1][0])
    if fail:
        ColorPrint("%s:%d - %s" % (caller.filename, caller.lineno, message), "FAIL")
    else:
        print("%s:%d - %s" % (caller.filename, caller.lineno, message))


class ProgressBar:

    BARLENGTH = 10

    def __init__(self,barLength):
        self.BARLENGTH = barLength # initialize the length of the progress bar

    def set_progress(self, progress):
        # update_progress() : Displays or updates a console progress bar
        ## Accepts a float between 0 and 1. Any int will be converted to a float.
        ## A value under 0 represents a 'halt'.
        ## A value at 1 or bigger represents 100%
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(self.BARLENGTH*progress))
        text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(self.BARLENGTH-block), progress*100, status)
        sys.stdout.write(bcolors.BOLDBLUE    + text + bcolors.ENDBOLD)
        sys.stdout.flush()
