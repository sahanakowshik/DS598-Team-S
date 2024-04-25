import numpy as np
import json
import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from unet_model import UNet
from transunet import TransUNet

EPISODE_DIR = (
    "./full_episodes/top_agents"
)
def parser():
    parser = argparse.ArgumentParser("pipeline", add_help=False)

    # Set Paths for running SSL training
    parser.add_argument('--model_dir', default='models', type=str,
        help='Please specify the model directory.')
    parser.add_argument('--model', default='unet', choices=['unet', 'transunet'], type=str,
        help='Please specify the model directory.')
    parser.add_argument('--multiplier', default=1, type=int,
        help='Please specify the model directory.')
    args = parser.parse_args()
    return args

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

args = parser()
print(args.model_dir)


MODEL_DIR = f"./{args.model_dir}/"
device = "cuda:1"
model_path = None

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        

seed = 42
seed_everything(seed)


def to_label(action, obs):
    strs = action.split(" ")
    unit_id = strs[1]
    if strs[0] == "m":
        label = {"n": 0, "s": 1, "w": 2, "e": 3}[strs[2]]
    elif strs[0] == "bcity":
        label = 4
    else:
        label = None

    unit_pos = (0, 0)

    width, height = obs["width"], obs["height"]
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    for update in obs["updates"]:
        strs = update.split(" ")
        if strs[0] == "u" and strs[3] == unit_id:
            unit_pos = (int(strs[4]) + x_shift, int(strs[5]) + y_shift)
    return unit_id, label, unit_pos


def depleted_resources(obs):
    for u in obs["updates"]:
        if u.split(" ")[0] == "r":
            return False
    return True


def create_dataset_from_json(episode_dir, team_names=["Toad Brigade"]):
    print("Creating datasets")
    obses = {}
    samples = []
    append = samples.append

    # get all .json files under the episode_dir even in subdirectories
    episodes = list(Path(episode_dir).rglob("*.json"))
    # clean episode files
    episodes = [
        str(ep) for ep in episodes if "info" not in str(ep) and "output" not in str(ep)
    ]

    # episodes = episodes[-1500:]
    # random.shuffle(episodes)

    for filepath in tqdm(episodes[:20]):
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load["info"]["EpisodeId"]
        index = np.argmax([r or 0 for r in json_load["rewards"]])
        if json_load["info"]["TeamNames"][index] not in team_names:
            continue

        for i in range(len(json_load["steps"]) - 1):
            try:
                if json_load["steps"][i][index]["status"] == "ACTIVE":
                    actions = json_load["steps"][i + 1][index]["action"]
                    obs = json_load["steps"][i][0]["observation"]

                    if depleted_resources(obs):
                        break

                    obs["player"] = index
                    obs = dict(
                        [
                            (k, v)
                            for k, v in obs.items()
                            if k in ["step", "updates", "player", "width", "height"]
                        ]
                    )

                    obs_id = f"{ep_id}_{i}"
                    obses[obs_id] = obs

                    # By initializing the map with -1, we can ignore positions where there is no friend worker
                    action_map = np.zeros((32, 32)) - 1

                    for action in actions:
                        unit_id, label, unit_pos = to_label(action, obs)
                        if label is not None:

                            action_map[unit_pos[0], unit_pos[1]] = label

                    append((obs_id, action_map))
            except:
                pass

    return obses, samples


def make_input(obs):
    width, height = obs["width"], obs["height"]
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    cities_opp = {}

    b = np.zeros((14, 32, 32), dtype=np.float32)
    b_global = np.zeros((15, 4, 4), dtype=np.float32)

    global_unit = 0
    global_rp = 0
    global_city = 0
    global_citytile = 0

    global_unit_opp = 0
    global_rp_opp = 0
    global_city_opp = 0
    global_citytile_opp = 0

    global_wood = 0
    global_coal = 0
    global_uranium = 0

    for update in obs["updates"]:
        strs = update.split(" ")
        input_identifier = strs[0]

        if input_identifier == "u":
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            team = int(strs[2])
            cooldown = float(strs[6])
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if team == obs["player"]:
                b[0, x, y] = 1  # b0 friend unit
                global_unit += 1
                b[1, x, y] = cooldown / 6  # b1 friend cooldown
                b[2, x, y] = (wood + coal + uranium) / 100  # b2 friend cargo
            else:
                b[3, x, y] = 1  # b3 oppo unit
                global_unit_opp += 1
                b[4, x, y] = cooldown / 6  # b4 oppo cooldown
                b[5, x, y] = (wood + coal + uranium) / 100  # b5 oppo cargo

        elif input_identifier == "ct":
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            if team == obs["player"]:
                global_citytile += 1
                b[6, x, y] = 1  # b6 friend city
                b[7, x, y] = cities[city_id]  # b7 friend city nights to survive
            else:
                global_citytile_opp += 1
                b[8, x, y] = 1  # b8 oppo city
                b[9, x, y] = cities_opp[city_id]  # b9 oppo city nights to survive
        elif input_identifier == "r":
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{"wood": 10, "coal": 11, "uranium": 12}[r_type], x, y] = amt / 800
            if r_type == "wood":
                global_wood += 1
            elif r_type == "coal":
                global_coal += 1
            else:
                global_uranium += 1
        elif input_identifier == "rp":
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            if team == obs["player"]:
                global_rp = min(rp, 200) / 200
            else:
                global_rp_opp = min(rp, 200) / 200
        elif input_identifier == "c":
            # Cities
            city_id = strs[2]
            team = int(strs[1])
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            if team == obs["player"]:
                global_city += 1
                cities[city_id] = min(fuel / lightupkeep, 20) / 20
            else:
                global_city_opp += 1
                cities_opp[city_id] = min(fuel / lightupkeep, 20) / 20
    # Map Size
    b[13, x_shift : 32 - x_shift, y_shift : 32 - y_shift] = 1
    # global features (normalized)
    b_global[0, :, :] = global_unit / width / height
    b_global[1, :, :] = global_rp
    b_global[2, :, :] = global_city / width / height
    b_global[3, :, :] = global_citytile / width / height
    b_global[4, :, :] = np.array(list(cities.values())).mean() if cities else 0
    b_global[5, :, :] = global_unit_opp / width / height
    b_global[6, :, :] = global_rp_opp
    b_global[7, :, :] = global_city_opp / width / height
    b_global[8, :, :] = global_citytile_opp / width / height
    b_global[9, :, :] = np.array(list(cities_opp.values())).mean() if cities_opp else 0
    b_global[10, :, :] = global_wood / width / height
    b_global[11, :, :] = global_coal / width / height
    b_global[12, :, :] = global_uranium / width / height
    b_global[13, :, :] = obs["step"] % 40 / 40  # Day/Night Cycle
    b_global[14, :, :] = obs["step"] / 360  # Turns

    return b, b_global


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, action_map = self.samples[idx]
        obs = self.obses[obs_id]
        state_1, state_2 = make_input(obs)

        return state_1, state_2, action_map


def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs):
    global best_acc
    global global_epoch
    global_epoch += 1

    for epoch in range(num_epochs):
        print("Learning with lr :", optimizer.state_dict()["param_groups"][0]["lr"])
        model.to(device)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0
            epoch_num = 0
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states_1 = item[0].to(device).float()
                states_2 = item[1].to(device).float()
                actions = item[2].to(device).long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    policy = model(states_1, states_2)

                    loss = criterion(policy, actions)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(actions == policy.argmax(dim=1))
                    # epoch_num is used to calculate the number of workers that actually go to loss, which can help us get the accuracy
                    epoch_num += torch.sum(actions >= 0)
            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / epoch_num

            print(
                f"Epoch {epoch}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
            )
        scheduler.step()

        if epoch_acc >= best_acc:
            print("save model...")
            traced = torch.jit.script(
                model.cpu(), (torch.rand(1, 14, 32, 32), torch.rand(1, 15, 4, 4))
            )
            traced.save(
                f"{MODEL_DIR}/model_{epoch_acc:.2f}.pth"
            )
            best_acc = epoch_acc


if __name__ == "__main__":
    # the directory of the training dataset
    episode_dir = EPISODE_DIR
    team_names = ["Toad Brigade", "RL is all you need", "skyramp", "Team Durrett", "ironbar"]
    obses, samples = create_dataset_from_json(episode_dir, team_names=team_names)
    print("obses:", len(obses), "samples:", len(samples))

    print(f"Using device: {device}")

    train, val = train_test_split(samples, test_size=0.1, random_state=42)
    batch_size = 512
    train_loader = DataLoader(
        LuxDataset(obses, train), batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        LuxDataset(obses, val), batch_size=batch_size, shuffle=False, num_workers=1
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    # We set ignore_index=-1 to ignore positions without workers
    
    if model_path:
        print("Loading existing model")
        model = torch.jit.load(model_path)
    else:
        print("Creating new model")
        if args.model.lower() == "unet":
            print("UNet")
            model = UNet(14, 5, 15, bilinear=True, multiplier=args.multiplier)
        elif args.model.lower() == "transunet":
            print("TransUNet")
            model = TransUNet(14, 5, 15, bilinear=True, multiplier=args.multiplier)
        else:
            raise UnknownModelTypeError
        
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), weight_decay=0.01)
    # Using exponential decayed LR
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer=optimizer,
    #         T_0=20,
    #         T_mult=2,
    #         eta_min = 0,
    #         verbose=False
    #     )
    best_acc = 0.0
    global_epoch = 0
    # for n in range(40):
    # print("Learning with lr :", optimizer.state_dict()["param_groups"][0]["lr"])
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=300)
    # We set the LR decayed every epoch
    # scheduler.step()
