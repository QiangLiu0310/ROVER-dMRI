import numpy as np 
import torch 
from torch.utils.data import Dataset

class ObservationPoints3D(Dataset):
    def __init__(self, *args, Y=None, batch_size=None):
        """
        Supports two call styles:
          1) ObservationPoints3D(*coords, Y=Y_flat, batch_size=batch_size)
          2) ObservationPoints3D(coord1, coord2, ..., Y_flat, batch_size)  # Legacy positional args
        """
        super().__init__()

        # If Y/batch_size provided as keywords, treat args as coords
        if Y is not None and batch_size is not None:
            coords = args
        else:
            # Otherwise assume last two positional args are Y and batch_size
            if len(args) < 3:
                raise ValueError("Need at least two positional args for Y and batch_size, or specify Y= and batch_size=")
            *coords, Y, batch_size = args

        # Convert to tuple
        self.coords = tuple(coords)
        if len(self.coords) == 0:
            raise ValueError("At least one coords array is required")

        self.Y = Y
        self.batch_size = int(batch_size)

        # Assume all coords have same first dimension N; record dimension info
        self.N = self.coords[0].shape[0]
        self.D = self.coords[0].shape[1]
        self.num_coords = len(self.coords)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rix = np.random.choice(self.N, size=self.batch_size, replace=False)
        coord_dict = {}
        for i, c in enumerate(self.coords):
            coord_dict[f"coord{i+1}"] = torch.from_numpy(c[rix, :])
        yvals = self.Y[rix, :]
        return coord_dict, {"yvals": torch.from_numpy(yvals)}

    def getfulldata(self):
        N = self.N
        coord_dict = {}
        for i, c in enumerate(self.coords):
            coord_dict[f"coord{i+1}"] = torch.from_numpy(c[0:N, :])
        return coord_dict, {"yvals": torch.from_numpy(self.Y[0:N, :])}