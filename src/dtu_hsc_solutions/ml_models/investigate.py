from argparse import ArgumentParser
from hsc_dataset import AudioDataset
from utils import create_data_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", default="1")
    parser.add_argument("--level", default="1")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    args = parser.parse_args()
    data_path = create_data_path(args.data_path, args.task, args.level)

    # Load your dataset
    dataset = AudioDataset(data_path)

    rec_longer= 0
    rec_same = 0

    for recorded_sig, clean_sig in dataset:
        # Check if the lengths of the recorded and clean signals
        if recorded_sig.shape > clean_sig.shape:
            rec_longer += 1
        elif recorded_sig.shape == clean_sig.shape:
            rec_same += 1

    print(f"Amount of recorded signals longer than clean signals: {rec_longer/len(dataset)*100:.2f}%")
    print(f"Amount of recorded signals same length as clean signals: {rec_same/len(dataset)*100:.2f}%")
            
