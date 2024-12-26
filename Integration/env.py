import gymnasium
from embedder_guesser import *
import torch.nn.functional as F
from gymnasium import spaces


class myEnv(gymnasium.Env):
    def __init__(self,
                 flags
                 ):
        super(myEnv, self).__init__()
        self.guesser = MultimodalGuesser()

        self.action_space = spaces.Discrete(self.guesser.tests_number + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.guesser.features_total,),
            dtype=np.float32
        )

        self.device = flags.device
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.guesser.X, self.guesser.y,
                                                                                test_size=0.1, random_state=42)
        self.cost_list = [1] * (self.guesser.tests_number + 1)
        self.prob_list = [cost / sum(self.cost_list) for cost in self.cost_list]
        self.cost_budget = flags.cost_budget
        self.num_classes = self.guesser.num_classes
        save_dir = os.path.join(os.getcwd(), flags.save_guesser_dir)
        guesser_filename = 'best_guesser.pth'
        guesser_load_path = os.path.join(save_dir, guesser_filename)
        if os.path.exists(guesser_load_path):
            print('Loading pre-trained guesser')
            guesser_state_dict = torch.load(guesser_load_path)
            self.guesser.load_state_dict(guesser_state_dict)

    def reset(self, seed=None, mode='training', patient=0):
        super().reset(seed=seed)  # This ensures compatibility with Gym
        self.state = np.zeros(self.guesser.features_total, dtype=np.float32)
        if seed is not None:
            np.random.seed(seed)
        if mode == 'training':
            self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        self.done = False
        self.total_cost = 0
        self.taken_actions = set()  # Reset the set of taken actions

        info = {}  # Add any relevant environment information here
        return self.state, info

    def step(self, action, mode='training'):
        # Convert action to scalar if needed
        if isinstance(action, torch.Tensor):
            action_number = int(action.item())
        else:
            action_number = action

        # Filter actions to exclude already-taken ones
        available_actions = [a for a in range(self.action_space.n) if a not in self.taken_actions]

        if action_number not in available_actions:
            print(f"Action {action_number} already taken. Choosing a new action.")
            action_number = np.random.choice(available_actions)  # Randomly choose from available actions

        # Mark the action as taken
        self.taken_actions.add(action_number)

        next_state = self.update_state(action_number, mode)
        self.total_cost += self.cost_list[action_number]
        self.state = np.array(next_state)
        reward = self._compute_internal_reward(mode)

        terminated = self.total_cost >= self.guesser.tests_number or self.done  # Episode ends naturally
        info = {'guess': self.guess}

        return self.state, reward, terminated, True, info

    def prob_guesser(self, state):
        guesser_input = torch.Tensor(
            state[:self.guesser.features_total])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        self.probs = self.guesser(guesser_input).squeeze()
        self.guess = torch.argmax(self.probs).item()
        self.correct_prob = self.probs[int(self.y_train[self.patient])].item()
        return self.correct_prob

    def prob_guesser_for_positive(self, state):
        guesser_input = torch.Tensor(
            state[:self.guesser.features_total])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        return self.guesser(guesser_input).squeeze()[1].item()

    def update_state(self, action, mode):
        prev_state = self.state
        next_state = np.array(self.state)
        if action < self.guesser.tests_number:  # Not making a guess
            features_revealed = self.guesser.map_test[action]
            for feature in features_revealed:
                if mode == 'training':
                    answer = self.X_train.iloc[self.patient, feature]
                elif mode == 'test':
                    answer = self.X_test.iloc[self.patient, feature]
                # check type of feature
                if self.is_numeric_value(answer):
                    answer_vec = torch.tensor([answer], dtype=torch.float32).unsqueeze(0)
                elif self.is_image_value(answer):
                    answer_vec = self.guesser.embed_image(answer)
                elif self.is_text_value(answer):
                    answer_vec = self.guesser.embed_text(answer).squeeze()
                else:
                    size = len(self.guesser.map_feature[feature])
                    answer_vec = [0] * size

                map_index = self.guesser.map_feature[feature]
                for count, index in enumerate(map_index):
                    next_state[index] = answer_vec[count]

            self.prob_classes = self.prob_guesser_for_positive(next_state)
            self.reward = abs(self.prob_guesser(next_state) - self.prob_guesser(prev_state)) / self.cost_list[action]
            self.guess = -1
            self.done = False
            return next_state

        else:
            self.prob_classes = self.prob_guesser_for_positive(prev_state)
            self.reward = self.prob_guesser(prev_state)
            self.done = True
            return prev_state

    def _compute_internal_reward(self, mode):
        """ Compute the reward """
        if mode == 'test':
            return None
        return self.reward

    def is_numeric_value(self, value):
        # Check if the value is an integer, a floating-point number, or a tensor of type float or double
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, torch.Tensor):
            if value.dtype in [torch.float, torch.float64]:
                return True
        return False

    def is_text_value(self, value):
        # Check if the value is a string
        if isinstance(value, str):
            return True
        else:
            return False

    def is_image_value(self, value):
        # check if value is path that ends with 'png' or 'jpg'
        if isinstance(value, str):
            if value.endswith('png') or value.endswith('jpg'):
                return True
            else:
                return False
