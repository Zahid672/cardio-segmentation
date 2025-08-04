import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        # self.load_state_dict(parameters)

        try:
            self.load_state_dict(parameters)
        except:
            print('Skipping layers whose parameter shapes do not match with pretrained weights')
            pretrained_dict = parameters
            model_dict = self.state_dict()
            
            # Filter out unnecessary keys
            same_size = 0
            all_size = 0
            for k, v in pretrained_dict.items():
                all_size += 1
                if k in model_dict and model_dict[k].shape == v.shape:
                    same_size += 1
                    
            print('Using only ', (same_size*100)/all_size, ' percent of pretrained weights')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
