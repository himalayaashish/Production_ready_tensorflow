from configs.config import CFG
from model.model import CLICK


def run():
    """Builds model, loads data, trains and evaluates"""
    model = CLICK(CFG)
    model.load_data()
    model.build()
    model.train()
    model.load_test_data()
    model.evaluate()


if __name__ == '__main__':
    run()