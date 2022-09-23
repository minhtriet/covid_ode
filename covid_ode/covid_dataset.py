from torch.utils.data import Dataset


class CovidDataset(Dataset):
    def __init__(self, data_frame, len_input, len_output):
        self.data = data_frame
        self.len_input = len_input
        self.len_output = len_output

    def __len__(self):
        return self.data.shape[0] - (self.len_input + self.len_output)

    def __getitem__(self, index):
        return self.data[index*self.len_input : (index+1)*self.len_input], \
               self.data[(index+1)*self.len_input : (index+1)*self.len_input+self.len_output]