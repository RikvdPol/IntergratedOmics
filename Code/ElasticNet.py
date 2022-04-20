class ElasticNet:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None

    def extract_labels(self):
        self.labels = self.file[self.labelname]
        print(self.labels)
git