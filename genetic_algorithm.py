import random
import cv2
import numpy as np
import os


def ensure_color_value_bounds(value):
    return min(255, max(0, value))


def random_color(is_grayscale):
    if is_grayscale:
        return int(random.randint(0, 255))
    else:
        return random_color(True), random_color(True), random_color(True)


def mutate_color(color, is_grayscale, color_mutation_intensity):
    assert color_mutation_intensity <= 1
    max_mutation = int(color_mutation_intensity*255)
    if is_grayscale:
        shift = random.randint(-max_mutation, max_mutation)
        return int(max(0, min(255, color + shift)))
    else:
        shift1 = random.randint(1, max_mutation)
        shift2 = random.randint(1, max_mutation)
        shift3 = random.randint(1, max_mutation)
        return ensure_color_value_bounds(color[0] + shift1), \
               ensure_color_value_bounds(color[1] + shift2), \
               ensure_color_value_bounds(color[2] + shift3)


def complement_color(color, is_grayscale):
    if is_grayscale:
        return int(255 - color)
    else:
        return complement_color(color[0], True), complement_color(color[1], True), complement_color(color[2], True)


def random_interpolation(color1, color2, is_grayscale):
    if not is_grayscale:
        return random_interpolation(color1[0], color2[0], True), \
               random_interpolation(color1[1], color2[1], True), \
               random_interpolation(color1[2], color2[2], True)
    else:
        min_c, max_c = min(color1, color2), max(color1, color2)
        return int(min_c + random.uniform(0.01, 0.99) * (max_c - min_c))


def mkdir_if_doesnt_exist(desired_folder_path):
    if not os.path.exists(desired_folder_path):
        path_chunks = desired_folder_path.split(os.sep)
        assert "." not in path_chunks[-1]
        cnt_chunk = path_chunks[0] + os.sep
        for chunk in path_chunks[1:]:
            cnt_chunk = os.path.join(cnt_chunk, chunk)
            if not os.path.exists(cnt_chunk):
                os.mkdir(cnt_chunk)


class Individual2D:

    CONFIG = None
    CHR_HEIGHT = None
    CHR_WIDTH = None
    IS_GRAYSCALE = None
    BACKGROUND = None
    INDIVIDUAL_CNT = 0

    def __init__(self, body, config=None, randomize=False):
        self.body = body
        self.fitness = 0
        Individual2D.IS_GRAYSCALE = len(self.body.shape) < 3
        if Individual2D.CONFIG is None:
            Individual2D.CONFIG = config
        ratio_mutation = Individual2D.CONFIG["MAX_SQUARES_TO_MUTATE_RATIO"]
        assert ratio_mutation >= 0 and ratio_mutation < 1
        Individual2D.set_rows_and_cols(Individual2D.CONFIG["N_ROWS"], Individual2D.CONFIG["N_COLUMNS"])
        Individual2D.CHR_HEIGHT = self.body.shape[0] // Individual2D.CONFIG["N_ROWS"]
        Individual2D.CHR_WIDTH = self.body.shape[1] // Individual2D.CONFIG["N_COLUMNS"]
        starting_color = Individual2D.CONFIG["STARTING_COLOR_INTENSITY"]
        Individual2D.BACKGROUND = (int(starting_color), int(starting_color), int(starting_color))
        if randomize:
            self.randomize()

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return abs(self.fitness - other.fitness) < 0.01

    def __add__(self, other):
        self.mutate_brightness(other)

    def __radd__(self, other):
        self + other

    def __pow__(self, power, modulo=None):
        self.power_mutation(power)

    def serialize(self):
        Individual2D.INDIVIDUAL_CNT += 1
        path = Individual2D.CONFIG["PATH"]
        cnt_epoch = Individual2D.CONFIG["CNT_EPOCH"]
        full_path = os.path.join(path, f"_epoch_{str(cnt_epoch).zfill(6)}")
        mkdir_if_doesnt_exist(full_path)
        fitness_str = f"{self.fitness}".replace(".", "Q")
        full_path = os.path.join(full_path,
                                 f"ind_{str(Individual2D.INDIVIDUAL_CNT).zfill(5)}" +
                                 f" fitness_{fitness_str}.jpg")
        cv2.imwrite(full_path, self.body)

    @staticmethod
    def deserialize(full_pth):
        body = cv2.imread(full_pth)
        img_name = full_pth.split(os.sep)[-1]
        fitness = float(img_name.split(" ")[-1].replace(".jpg", "").replace("fitness_", "").replace("Q", "."))
        ind = Individual2D(body)
        ind.fitness = fitness
        return ind

    def randomize(self):
        w, h = Individual2D.CHR_WIDTH, Individual2D.CHR_HEIGHT
        for cx in range(Individual2D.CONFIG["N_COLUMNS"]):
            for cy in range(Individual2D.CONFIG["N_ROWS"]):
                pt1 = int(w * cx), int(h * cy)
                pt2 = int(min(w * (cx + 1), self.body.shape[1])), int(min(h * (cy + 1), self.body.shape[0]))
                intensity = Individual2D.CONFIG["STARTING_COLOR_INTENSITY"]
                color = (int(intensity), int(intensity), int(intensity))
                color = random_color(Individual2D.IS_GRAYSCALE) if random.randint(0, 100) < 95 else color
                if random.uniform(0, 1) < 0.8:
                    cv2.rectangle(self.body, pt1, pt2, thickness=-1, color=color)
                else:
                    xs, ys = int((pt1[0] + pt2[0])/2.0), int((pt1[1] + pt2[1])/2.0)
                    random_radius = abs(int((pt2[0] - pt1[0])/2.0))
                    if random.randint(0, 1):
                        random_radius += random_radius * random.uniform(-0.5, 0.5)
                    random_radius = int(random_radius)
                    cv2.circle(self.body, (xs, ys), random_radius, color, -1)

    def shape(self):
        return self.body.shape

    def _prepare_colors_for_crossover(self, other, pt):
        col1 = self.body[pt[1], pt[0]]
        col2 = other.body[pt[1], pt[0]]
        if random.randint(0, 1):
            col1 = random_interpolation(col1, col2, Individual2D.IS_GRAYSCALE)
            col2 = random_interpolation(col1, col2, Individual2D.IS_GRAYSCALE)
        col1 = (int(col1[0]), int(col1[1]), int(col1[2])) if not Individual2D.IS_GRAYSCALE \
            else (int(col1), int(col1), int(col1))
        col2 = (int(col2[0]), int(col2[1]), int(col2[2])) if not Individual2D.IS_GRAYSCALE \
            else (int(col2), int(col2), int(col2))
        return col1, col2

    def _whole_interpolation_crossover(self, other):
        rand_ratio = random.uniform(0, 1)
        child_body_1 = self.body.astype("float") * rand_ratio + other.body.astype("float") * (1.0 - rand_ratio)
        child_body_2 = other.body.astype("float") * rand_ratio + self.body.astype("float") * (1.0 - rand_ratio)
        child_body_1 = child_body_1.astype("uint8")
        child_body_2 = child_body_2.astype("uint8")
        return child_body_1, child_body_2

    def _line_crossover(self, other, line_type="horizontal"):
        child_body_1 = self.body.copy()
        child_body_2 = self.body.copy()
        if random.randint(0, 1):
            w, h = Individual2D.CHR_WIDTH, Individual2D.CHR_HEIGHT
            n_steps = Individual2D.CONFIG["N_ROWS"] if line_type == "horizontal" else Individual2D.CONFIG["N_COLUMNS"]
        else:
            random_ratio = random.uniform(0.05, 0.35)
            w, h = self.shape()[1]*random_ratio, self.shape()[0]*random_ratio
            n_steps = Individual2D.CONFIG["N_ROWS"] if line_type == "horizontal" else Individual2D.CONFIG["N_COLUMNS"]
        for cnt in range(n_steps):
            if line_type == "horizontal":
                pt1 = 0, min(h * cnt, self.shape()[0])
                pt2 = self.shape()[1], min(h * (cnt + 1), self.body.shape[0])
            else:
                pt1 = max(w * cnt, self.shape()[1]), 0
                pt2 = min(w * (cnt + 1), self.body.shape[1]), self.shape()[0]
            pt1, pt2 = (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))
            if pt1[1] == self.shape()[1] or pt1[0] == self.shape()[0]:
                break
            if random.randint(0, 1):
                child_body_1[pt1[1]:pt2[1], pt1[0]:pt2[0]] = self.body[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                child_body_2[pt1[1]:pt2[1], pt1[0]:pt2[0]] = other.body[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            else:
                child_body_1[pt1[1]:pt2[1], pt1[0]:pt2[0]] = other.body[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                child_body_2[pt1[1]:pt2[1], pt1[0]:pt2[0]] = self.body[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        return child_body_1, child_body_2

    def _square_crossover(self, other):
        child_body_1 = self.body.copy()
        child_body_2 = self.body.copy()
        w, h = Individual2D.CHR_WIDTH, Individual2D.CHR_HEIGHT
        for cx in range(Individual2D.CONFIG["N_COLUMNS"]):
            for cy in range(Individual2D.CONFIG["N_ROWS"]):
                pt1 = w * cx, h * cy
                pt2 = min(w * (cx + 1), self.body.shape[1]), min(h * (cy + 1), self.body.shape[0])
                col1, col2 = self._prepare_colors_for_crossover(other, pt1)
                self._paint_children(child_body_1, child_body_2, pt1, pt2, col1, col2)
        return child_body_1, child_body_2

    def crossover_with(self, other):
        child_body_1 = self.body.copy()
        child_body_2 = self.body.copy()
        determine = random.randint(0, 3)
        if determine == 0:
            child_body_1, child_body_2 = self._square_crossover(other)
        elif determine == 1:
            child_body_1, child_body_2 = self._whole_interpolation_crossover(other)
        elif determine == 2:
            child_body_1, child_body_2 = self._line_crossover(other, "horizontal")
        elif determine == 3:
            child_body_1, child_body_2 = self._line_crossover(other, "vertical")

        return Individual2D(child_body_1), Individual2D(child_body_2)

    def _paint_children(self, child_body_1, child_body_2, pt1, pt2, col1, col2):
        if random.randint(0, 1):
            cv2.rectangle(child_body_1, pt1, pt2, thickness=-1, color=col1)
            cv2.rectangle(child_body_2, pt1, pt2, thickness=-1, color=col2)
        else:
            cv2.rectangle(child_body_1, pt1, pt2, thickness=-1, color=col2)
            cv2.rectangle(child_body_2, pt1, pt2, thickness=-1, color=col1)

    def get_random_rect(self, discrete=True):
        if discrete:
            cx = random.randint(0, Individual2D.CONFIG["N_COLUMNS"] - 1)
            cy = random.randint(0, Individual2D.CONFIG["N_ROWS"] - 1)
            w, h = Individual2D.CHR_WIDTH, Individual2D.CHR_HEIGHT
            pt1 = w * cx, h * cy
            pt2 = min(w * (cx + 1), self.body.shape[1]), min(h * (cy + 1), self.body.shape[0])
            return pt1, pt2
        else:
            x0, y0 = random.randint(0, self.shape()[1]-1), random.randint(0, self.shape()[0]-1)
            smaller_dimension = min(self.shape()[0], self.shape()[1])
            max_shape_size = smaller_dimension * Individual2D.CONFIG["SHAPE_BIGGEST_SIZE_PERCENTAGE"]
            rand_width, rand_height = random.randint(2, max_shape_size), random.randint(2, max_shape_size)
            x_s, y_s = max(0, x0 - rand_width/2), max(0, y0 - rand_height/2)
            x_e, y_e = min(self.shape()[1], x0 + rand_width / 2), min(self.shape()[0], y0 + rand_height / 2)
            return (int(x_s), int(y_s)), (int(x_e), int(y_e))

    def mutate(self):
        determine = random.randint(0, 4)
        if determine == 0:
            self.squared_mutation(random.uniform(0, 1) < Individual2D.CONFIG["COLOR_COMPLEMENT_CHANCE"])
        elif determine == 1:
            self.random_insert_mutation(random.uniform(0, 1) < Individual2D.CONFIG["COLOR_COMPLEMENT_CHANCE"])
        elif determine == 2:
            self.power_mutation()
        elif determine == 3:
            self.mutate_brightness()

    def squared_mutation(self, complemented_color=False):
        n_squares = Individual2D.CONFIG["N_COLUMNS"] * Individual2D.CONFIG["N_ROWS"]
        repeat = random.randint(1, int(Individual2D.CONFIG["MAX_SQUARES_TO_MUTATE_RATIO"] * n_squares))
        done = set()
        while len(done) < repeat:
            pt1, pt2 = self.get_random_rect()
            size = len(done)
            done.add((pt1, pt2))
            if size < len(done):
                if not random.randint(0, 4):
                    w = int(Individual2D.CHR_WIDTH*random.uniform(0.05, 0.2))
                    h = int(Individual2D.CHR_HEIGHT*random.uniform(0.05, 0.2))
                    if w > 2 and h > 2:
                        pt_cnt_x = pt1[0]
                        while pt_cnt_x + w < pt2[0]:
                            pt_cnt_y = pt1[1]
                            while pt_cnt_y + h < pt2[1]:
                                if not random.randint(0, 3):
                                    pts = (int(pt_cnt_x), int(pt_cnt_y))
                                    pte = (int(pt_cnt_x+w), int(pt_cnt_y+h))
                                    self._squared_mutation(pts, pte, w, h, complemented_color)
                                pt_cnt_y += h
                            pt_cnt_x += w
                else:
                    w, h = Individual2D.CHR_WIDTH, Individual2D.CHR_HEIGHT
                    if not self._squared_mutation(pt1, pt2, w, h, complemented_color):
                        done.remove((pt1, pt2))

    def _squared_mutation(self, pt1, pt2, w, h, complemented_color=False):
        if random.randint(0, 1):
            new_color = mutate_color(
                self.body[pt1[1], pt1[0]],
                Individual2D.IS_GRAYSCALE,
                Individual2D.CONFIG["COLOR_MUTATION_INTENSITY"]
            )
        else:
            shift_x = random.randint(-1, 1)
            shift_y = ((-1) ** random.randint(0, 1)) if shift_x == 0 else 0
            try:
                color1, color2 = self.body[pt1[1], pt1[0]], self.body[
                    pt1[1] + shift_y * w, pt1[0] + shift_x * h]
                new_color = random_interpolation(color1, color2, Individual2D.IS_GRAYSCALE)
            except:
                return False
        color = (int(new_color[0]), int(new_color[1]), int(new_color[2])) if not Individual2D.IS_GRAYSCALE else \
            int(new_color)
        if complemented_color:
            color = complement_color(color, Individual2D.IS_GRAYSCALE)

        cv2.rectangle(
            self.body, pt1, pt2, thickness=-1, color=color
        )
        return True

    def _draw_random_circle(self, complemented_color=False):
        pt1, pt2 = self.get_random_rect(random.randint(0, 1) == 1)
        random_radius = pt2[1] - pt1[1]
        xs, ys, = int((pt1[0] + pt2[0])/2), int((pt1[1] + pt2[1])/2)
        rand_color = random_color(Individual2D.IS_GRAYSCALE)
        rand_color = random_interpolation(rand_color, self.body[pt1[1], pt1[0]], Individual2D.IS_GRAYSCALE)
        if complemented_color:
            rand_color = complement_color(rand_color, Individual2D.IS_GRAYSCALE)
        cv2.circle(self.body, (xs, ys), random_radius, rand_color, -1)

    def _draw_random_rect(self, complemented_color=False):
        pt1, pt2 = self.get_random_rect(random.randint(0, 1) == 1)
        rand_color = random_color(Individual2D.IS_GRAYSCALE)
        rand_color = random_interpolation(rand_color, self.body[pt1[1], pt1[0]], Individual2D.IS_GRAYSCALE)
        if complemented_color:
            rand_color = complement_color(rand_color, Individual2D.IS_GRAYSCALE)
        cv2.rectangle(self.body, pt1, pt2, rand_color, -1)

    def random_insert_mutation(self, complemented_color=False):
        determine = random.randint(0, 1)
        if determine == 0:
            self._draw_random_rect(complemented_color)
        elif determine == 1:
            self._draw_random_circle(complemented_color)

    def power_mutation(self, value=None):
        self.body = self.body.astype('float')
        self.body *= (1.0 / 255.0)
        self.body **= value if value is not None else random.uniform(0.1, 3)
        self.body *= 255
        self.body = self.body.astype('uint8')

    def mutate_brightness(self, value=None):
        self.body = self.body.astype('float')
        if value is None:
            diff = int(255*Individual2D.CONFIG["BRIGHTNESS_MUTATION_INTENSITY"])
            self.body += random.randint(-diff, diff)
        else:
            self.body += value
        self.body = np.clip(self.body, 0, 255)
        self.body = self.body.astype("uint8")

    @staticmethod
    def set_rows_and_cols(rows, cols):
        img = Individual2D.CONFIG["IMAGE"]
        Individual2D.CONFIG["N_ROWS"] = rows
        Individual2D.CONFIG["N_COLUMNS"] = cols
        Individual2D.CHR_HEIGHT = img.shape[0] // Individual2D.CONFIG["N_ROWS"]
        Individual2D.CHR_WIDTH = img.shape[1] // Individual2D.CONFIG["N_COLUMNS"]


class GeneticAlgorithm:

    def __init__(self, config, population=None):
        self.config = config
        self.config["CNT_EPOCH"] = 0
        self._ensure_image_channels()
        self.population = [
            Individual2D(np.zeros(config["IMAGE"].shape, int), config, randomize=True)
            for _ in range(config["POPULATION_SIZE"])
        ] if population is None else population

    def __getitem__(self, item):
        return self.population[item]

    def __len__(self):
        return len(self.population)

    def _ensure_image_channels(self):
        if self.config["IS_GRAYSCALE"] and len(self.config["IMAGE"].shape) == 3:
            self.config["IMAGE"] = cv2.cvtColor(self.config["IMAGE"], cv2.COLOR_BGR2GRAY)

    def serialize(self):
        [ind.serialize() for ind in self.population]

    @staticmethod
    def deserialize(config):
        folders = sorted([name for name in os.listdir(config["PATH"]) if "." not in name])
        assert len(folders) > 0
        population_path = os.path.join(config["PATH"], folders[-1])
        image_names = [name for name in os.listdir(population_path) if name.endswith(".jpg")]
        loaded_population = []
        for image_name in image_names:
            image_full_path = os.path.join(population_path, image_name)
            loaded_image = cv2.imread(image_full_path)
            if config["IS_GRAYSCALE"]:
                loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
            loaded_population.append(Individual2D(loaded_image, config))
        ga_instance = GeneticAlgorithm(config, loaded_population)
        ga_instance.config["CNT_EPOCH"] = int(folders[-1].split("_")[-1])
        ga_instance._ensure_image_channels()
        return ga_instance

    def fitness_function(self, individual):
        n_pixels = Individual2D.CONFIG["IMAGE"].shape[0] * Individual2D.CONFIG["IMAGE"].shape[1]
        diff = individual.body.astype("float") - Individual2D.CONFIG["IMAGE"].astype("float")
        return np.linalg.norm(
            diff
        ) / float(n_pixels)

    def calculate_overall_fitness_and_sort(self):
        for ind in self.population:
            ind.fitness = self.fitness_function(ind)
        self.population.sort()

    def reproduce_and_mutate(self):
        the_best = self[0]
        second_parent = self[1] if random.randint(0, 100) < 98 else self[random.randint(1, len(self)-1)]
        self.population[1] = second_parent
        for i in range(1, len(self)//2):
            rs, cs = Individual2D.CONFIG["N_ROWS"], Individual2D.CONFIG["N_COLUMNS"]
            if i > 100 and not random.randint(0, 3):
                Individual2D.set_rows_and_cols(
                    Individual2D.CONFIG["N_ROWS"] * random.uniform(1, 4),
                    Individual2D.CONFIG["N_COLUMNS"] * random.uniform(1, 4)
                )
            child1, child2 = the_best.crossover_with(second_parent)
            old_survives = random.uniform(0, 1) < Individual2D.CONFIG["OLD_SURVIVAL_RATE"]
            self.population[2*i] = self.population[2*i] if old_survives else child1
            old_survives = random.uniform(0, 1) < Individual2D.CONFIG["OLD_SURVIVAL_RATE"]
            self.population[2*i+1] = self.population[2*i+1] if old_survives else child2
            if random.uniform(0, 1) < Individual2D.CONFIG["MUTATION_RATE"]:
                self.population[2*i].mutate()
            if random.uniform(0, 1) < Individual2D.CONFIG["MUTATION_RATE"]:
                self.population[2*i+1].mutate()
            Individual2D.set_rows_and_cols(rs, cs)
        if random.uniform(0, 1) < Individual2D.CONFIG["MUTATE_PARENTS_CHANCE"]:
            self[0].mutate()
            self[1].mutate()

    def run(self):
        self.current_epoch = 0
        start_from = Individual2D.CONFIG["CNT_EPOCH"] if "CNT_EPOCH" in Individual2D.CONFIG else 0
        for i in range(start_from, Individual2D.CONFIG["N_EPOCHS"]):
            self.calculate_overall_fitness_and_sort()
            self.current_epoch = i+1
            self.config["CNT_EPOCH"] = self.current_epoch
            if i % Individual2D.CONFIG["INCREASE_DIMENSION_EVERY_NTH_EPOCH"] == 0 \
                    and min(Individual2D.CHR_WIDTH, Individual2D.CHR_HEIGHT) > 5:
                Individual2D.set_rows_and_cols(
                    Individual2D.CONFIG["N_ROWS"] + 1,
                    Individual2D.CONFIG["N_COLUMNS"] + 1
                )
            if (i+1) % 100 == 0:
                print(f"EPOCH {self.current_epoch}")
            if (i+1) % Individual2D.CONFIG["SAVE_EVERY_NTH_PICTURE"] == 0:
                self.save_the_best()
            if (i+1) % Individual2D.CONFIG["SERIALIZE_EVERY_NTH_EPOCH"] == 0 or i == 0:
                self.serialize()
            self.reproduce_and_mutate()

    def save_the_best(self):
        to_save = self[0].body
        full_path = os.path.join(Individual2D.CONFIG["PATH"], f"the_best_{f'{self.current_epoch}'.zfill(5)}.jpg")
        cv2.imwrite(full_path, to_save)


if __name__ == "__main__":
    image = cv2.imread("./sample_image.jpeg")
    CONFIG = {
        "N_EPOCHS": 10000,
        "POPULATION_SIZE": 100,
        "VALUE_BOUNDS": 20,
        "N_ROWS": 20,
        "N_COLUMNS": 20,
        "MAX_SQUARES_TO_MUTATE_RATIO": 0.35,
        "BRIGHTNESS_MUTATION_INTENSITY": 0.1,
        "COLOR_MUTATION_INTENSITY": 0.35,
        "SHAPE_BIGGEST_SIZE_PERCENTAGE": 0.2,
        "MUTATION_RATE": 0.6,
        "STARTING_COLOR_INTENSITY": 0,
        "SAVE_EVERY_NTH_PICTURE": 1,
        "SERIALIZE_EVERY_NTH_EPOCH": 10,
        "INCREASE_DIMENSION_EVERY_NTH_EPOCH": 25,
        "MUTATE_PARENTS_CHANCE": 0.1,
        "COLOR_COMPLEMENT_CHANCE": 0.05,
        "OLD_SURVIVAL_RATE": 0.05,
        "PATH": "./saved",
        "IMAGE": image,
        "IS_GRAYSCALE": True
    }
    GeneticAlgorithm(CONFIG).run()
