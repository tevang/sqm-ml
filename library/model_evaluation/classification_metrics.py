import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

class Classification_Metric():

    def __init__(self, y, x):
        """

        :param y: iterable of true labels, which must be 0 or 1 only! No multi-label is supported in this class!
        :param x: iterable of predicted labels, 0 or 1.
        """
        mcm = multilabel_confusion_matrix(y, x)
        self.tp = mcm[0, 0, 0]
        self.tn = mcm[0, 1, 1]
        self.fp = mcm[0, 1, 0]
        self.fn = mcm[0, 0, 1]

    def TPR(self):
        """
        True Positive rate (TPR) aka Sensitivity.
        :param x:
        :param y:
        :return:
        """
        return self.tp/(self.tp+self.fn)

    def FPR(self):
        """
        False Positive Rate (FPR)
        :param x:
        :param y:
        :return:
        """
        return 1-self.TNR()
    
    def TNR(self):
        """
        True Negative Rate (TNR) aka Specificity.
        :param x:
        :param y:
        :return:
        """
        return self.tn/(self.tn+self.fp)
    
    def FNR(self):
        """
        False Negative Rate (FNR).
        :param x:
        :param y:
        :return:
        """
        return 1-self.TPR()

    def LRplus(self):
        """
        Positive likelihood ratio (LR+).
        :param x:
        :param y:
        :return:
        """
        return self.TPR()/self.FPR()
    
    def LRminus(self):
        """
        Negative likelihood ratio (LR−).
        :param x:
        :param y:
        :return:
        """
        return self.FNR()/self.TNR()
    
    def DOR(self):
        """
        Diagnostic odds ratio (DOR).
        The diagnostic odds ratio ranges from zero to infinity, although for useful tests it is
        greater than one, and higher diagnostic odds ratios are indicative of better test performance.
        Diagnostic odds ratios less than one indicate that the test can be improved by simply inverting
        the outcome of the test – the test is in the wrong direction, while a diagnostic odds ratio of
        exactly one means that the test is equally likely to predict a positive outcome whatever the
        true condition – the test gives no information.
        :param x:
        :param y:
        :return:
        """
        return self.LRplus()/self.LRminus()

    def PPV(self):
        """
        Positive predictive value (PPV) aka Precision.
        :return:
        """
        return self.tp/(self.tp+self.fp)

    def NPV(self):
        """
        Negative predictive value (NPV).
        :return:
        """
        return self.tn/(self.tn+self.fn)

    def MK(self):
        """
        Markedness (MK).
        :return:
        """
        return self.PPV()+self.NPV()-1
