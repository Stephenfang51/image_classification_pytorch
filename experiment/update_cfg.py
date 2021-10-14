import yaml
import os
from datetime import date
from torch.cuda import device_count

def decor_partingline(fn):
    def parting_line(arg):
        print('*'*50)
        return fn(arg)
    return parting_line


@decor_partingline
def update_cfg(args):
    """
    args : from argparser
    """
    assert os.path.exists(args.config), f"cannot found args.cfg : {args.config}"
    with open(args.config, errors='ignore') as file:
        dict_yaml = yaml.safe_load(file)
        #check if batchsize changed through arg parser
        if args.batch_size is not None:
            print(f'updating batch_size to {args.batch_size}')
            assert 16 < args.batch_size < 512, "please set batchsize range from 0 to 256 !"
            dict_yaml['train_batchsize'] = args.batch_size
        if args.gpus is not None:
            print(f'updating gpus to {args.gpus}')
            device_num = device_count()
            assert 1 < int(args.gpus[-1])+1 <= device_num, f"your system got {device_num} GPUS, now you set {int(args.gpus[-1])+1}!"
            dict_yaml['gpus'] = args.gpus

    [print(key, value) for key, value in dict_yaml.items()]

    return dict_yaml

@decor_partingline
def create_dir(dict_cfg):
    #create output_model/2021xxxx_{MODEL_NAME}/1 directorys
    today = date.today()
    date_format = today.strftime("%b-%d-%Y")
    if os.path.exists("{}/{}_{}".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"])) == False:
        os.makedirs("{}/{}_{}/{}".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"], 1))
        with open("{}/{}_{}/1/training_config.yaml".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"]), "w") as file:
            config_log = yaml.dump(dict_cfg, file)
        save_path = "{}/{}_{}/1".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"])
    else:
        max_num = max([int(directory) for directory in os.listdir("{}/{}_{}".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"]))]) + 1
        os.makedirs("{}/{}_{}/{}".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"], max_num), exist_ok=True)
        with open("{}/{}_{}/{}/training_config.yaml".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"], max_num), "w") as file:
            config_log = yaml.dump(dict_cfg, file)
        # save_path = "./output_model/{}_{}/{}".format(date_format, dict_cfg["model_arch"], max_num)
        save_path = "{}/{}_{}/{}".format(dict_cfg["default_save_path"], date_format, dict_cfg["model_arch"], max_num)

    print(f'create {save_path} directory' )
    return save_path

        