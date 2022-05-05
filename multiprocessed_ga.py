from genetic_algorithm import *
from multiprocessing import Process


def job(image, process_index):
    CONFIG = {
        "N_EPOCHS": 10000,
        "POPULATION_SIZE": 100,
        "VALUE_BOUNDS": 20,
        "N_ROWS": 4,
        "N_COLUMNS": 4,
        "MAX_SQUARES_TO_MUTATE_RATIO": 0.3,
        "BRIGHTNESS_MUTATION_INTENSITY": 0.05,
        "COLOR_MUTATION_INTENSITY": 0.20,
        "SHAPE_BIGGEST_SIZE_PERCENTAGE": 0.15,
        "MUTATION_RATE": 0.55,
        "STARTING_COLOR_INTENSITY": 255,
        "SAVE_EVERY_NTH_PICTURE": 5,
        "SERIALIZE_EVERY_NTH_EPOCH": 5,
        "INCREASE_DIMENSION_EVERY_NTH_EPOCH": 25,
        "MUTATE_PARENTS_CHANCE": 0.1,
        "COLOR_COMPLEMENT_CHANCE": 0.05,
        "OLD_SURVIVAL_RATE": 0.05,
        "PATH": f"C:\\Users\\a774880\\Desktop\\serialized\\serialized_{process_index}",
        "IMAGE": image,
        "IS_GRAYSCALE": True
    }

    GeneticAlgorithm(CONFIG).run()


if __name__ == "__main__":
    img = cv2.imread("C:\\Users\\a774880\\Desktop\\img.jpg")

    ps = [Process(target=job, args=(img[:, :, i], i)) for i in range(3)]
    [p.start() for p in ps]
    [p.join() for p in ps]
