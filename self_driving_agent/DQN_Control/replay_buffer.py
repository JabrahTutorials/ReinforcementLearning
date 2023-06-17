import cv2
import torch
import numpy as np
from torchvision import transforms

class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device) -> None:
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size,) + state_dim)
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).unsqueeze(1).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).unsqueeze(1).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

def test_buffer():
    img0 = np.zeros((5, 5))
    img1 = img0 + 1
    img2 = img0 + 2
    img3 = img0 + 3

    action = 1
    reward = 10
    done = 0

    device = "cpu"

    buffer = ReplayBuffer((5, 5), 2, 10, device)
    buffer.add(img0, action, img1, reward, done)
    buffer.add(img1, action, img2, reward, done)
    buffer.add(img2, action, img3, reward, done + 1)

    sample = buffer.sample()[0]
    print(sample.shape)

    norm = transforms.Normalize((0.5, 0.5), (0.5, 0.5))
    print(norm(sample).shape)

    

# test_buffer()
