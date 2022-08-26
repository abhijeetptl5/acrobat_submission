from torch.utils.data import Dataset

class WSIDataset(Dataset):
    def __init__(self, df, wsi, transform, level=0, ps=256):
        self.wsi = wsi
        self.transform = transform
        self.df = df
        self.level = level
        self.ps = ps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        offset = self.ps//2
        x, y = self.df.iloc[idx, 0]-offset, self.df.iloc[idx, 1]-offset
        if x<0: x=0
        if y<0: y=0
        patch = self.wsi.read_region((x, y), self.level, (self.ps, self.ps))
        patch = self.transform(patch.convert('RGB'))
        return patch
