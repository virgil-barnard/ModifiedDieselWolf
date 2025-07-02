# pytorch transforms for RF IQ data.
import numpy as np
import torch


class CarrierPhase(object):  # carrier phase offset transform.
    def __init__(self, theta, batch=False, data_keys=False):
        self.theta = theta
        self.rads = np.pi * theta / (180)
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            item["metadata"]["CarrierPhaseOffset"] = self.theta

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                rot = torch.tensor(
                    [
                        [np.cos(self.rads), np.sin(self.rads)],
                        [-np.sin(self.rads), np.cos(self.rads)],
                    ]
                )
                tensor_prime = torch.mm(rot, tensor)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    rot = torch.tensor(
                        [
                            [np.cos(self.rads), np.sin(self.rads)],
                            [-np.sin(self.rads), np.cos(self.rads)],
                        ]
                    )
                    tensor_prime = torch.mm(rot, tensor)
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                item["metadata"]["CarrierPhaseOffset"][idx] = self.theta

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    rot = torch.tensor(
                        [
                            [np.cos(self.rads), np.sin(self.rads)],
                            [-np.sin(self.rads), np.cos(self.rads)],
                        ]
                    )
                    tensor_prime = torch.mm(rot, tensor)
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        rot = torch.tensor(
                            [
                                [np.cos(self.rads), np.sin(self.rads)],
                                [-np.sin(self.rads), np.cos(self.rads)],
                            ]
                        )
                        tensor_prime = torch.mm(rot, tensor)
                        item[key][idx] = tensor_prime

        return item


class RandomCarrierPhase(object):  # random carrier phase offset transform.
    def __init__(
        self,
        theta_min=0,
        theta_max=360,
        batch=False,
        data_keys=False,
        rand_type="uniform",
    ):
        self.theta = 0
        self.rads = np.pi * self.theta / (180)
        self.theta_min = theta_min
        self.theta_max = theta_max
        if rand_type == "uniform":
            self.rng = torch.distributions.uniform.Uniform(
                self.theta_min, self.theta_max
            )
        elif rand_type == "normal":
            self.rng = torch.distributions.normal.Normal(self.theta_min, self.theta_max)
        else:
            print("No")
        self.batch = batch
        self.keys = data_keys
        self.rand_type = rand_type

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            self.theta = self.rng.sample().item()
            self.rads = np.pi * self.theta / (180)
            item["metadata"]["CarrierPhaseOffset"] = self.theta

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                rot = torch.tensor(
                    [
                        [np.cos(self.rads), np.sin(self.rads)],
                        [-np.sin(self.rads), np.cos(self.rads)],
                    ]
                ).float()
                tensor_prime = torch.mm(rot, tensor)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    rot = torch.tensor(
                        [
                            [np.cos(self.rads), np.sin(self.rads)],
                            [-np.sin(self.rads), np.cos(self.rads)],
                        ]
                    ).float()
                    tensor_prime = torch.mm(rot, tensor)
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                self.theta = self.rng.sample().item()
                self.rads = np.pi * self.theta / (180)
                item["metadata"]["CarrierPhaseOffset"][idx] = self.theta

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    rot = torch.tensor(
                        [
                            [np.cos(self.rads), np.sin(self.rads)],
                            [-np.sin(self.rads), np.cos(self.rads)],
                        ]
                    ).float()
                    tensor_prime = torch.mm(rot, tensor)
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        rot = torch.tensor(
                            [
                                [np.cos(self.rads), np.sin(self.rads)],
                                [-np.sin(self.rads), np.cos(self.rads)],
                            ]
                        ).float()
                        tensor_prime = torch.mm(rot, tensor)
                        item[key][idx] = tensor_prime

        return item


class CarrierFrequency(object):  # carrier frequency + phase offset transform.
    def __init__(self, delta_f, theta, batch=False, data_keys=False):
        self.theta = theta
        self.rads = np.pi * self.theta / (180)
        self.df = delta_f
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            item["metadata"]["CarrierPhaseOffset"] = self.theta
            item["metadata"]["CarrierFrequencyOffset"] = self.df

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                times = item["metadata"]["dt"] * np.arange(0, len(tensor[0]), 1).astype(
                    np.float32
                )
                x = self.rads + 2 * np.pi * self.df * times
                rot = torch.tensor([[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]])
                tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    times = item["metadata"]["dt"] * np.arange(
                        0, len(tensor[0]), 1
                    ).astype(np.float32)
                    x = self.rads + 2 * np.pi * self.df * times
                    rot = torch.tensor(
                        [[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]]
                    )
                    tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                item["metadata"]["CarrierPhaseOffset"][idx] = self.theta
                item["metadata"]["CarrierFrequencyOffset"][idx] = self.df

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    times = item["metadata"]["dt"][idx].item() * np.arange(
                        0, len(tensor[0]), 1
                    ).astype(np.float32)
                    x = self.rads + 2 * np.pi * self.df * times
                    rot = torch.tensor(
                        [[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]]
                    )
                    tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        times = item["metadata"]["dt"][idx].item() * np.arange(
                            0, len(tensor[0]), 1
                        ).astype(np.float32)
                        x = self.rads + 2 * np.pi * self.df * times
                        rot = torch.tensor(
                            [[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]]
                        )
                        tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                        item["data"][idx] = tensor_prime

        return item


class RandomCarrierFrequency(
    object
):  # random carrier frequency + phase offset transform.
    def __init__(
        self,
        delta_f_range,
        theta_min=0,
        theta_max=360,
        batch=False,
        data_keys=False,
        rand_type="uniform",
    ):
        self.theta = 0
        self.rads = np.pi * self.theta / (180)
        self.df_range = (
            delta_f_range  # in Hz, if data_rate is 1, in percent of data_rate
        )
        self.theta_min = theta_min
        self.theta_max = theta_max
        if rand_type == "uniform":
            self.rng_theta = torch.distributions.uniform.Uniform(
                self.theta_min, self.theta_max
            )
            self.rng_df = torch.distributions.uniform.Uniform(
                -self.df_range, self.df_range
            )
        elif rand_type == "normal":
            self.rng_theta = torch.distributions.normal.Normal(
                self.theta_min, self.theta_max
            )
            self.rng_df = torch.distributions.normal.Normal(
                -self.df_range, self.df_range
            )
        else:
            print("No")
        self.batch = batch
        self.keys = data_keys
        self.rand_type = rand_type

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            self.theta = self.rng_theta.sample().item()
            self.rads = np.pi * self.theta / (180)
            self.df = (
                self.rng_df.sample().item()
            )  # in Hz, if data_rate is 1, in percent of data_rate
            item["metadata"]["CarrierPhaseOffset"] = self.theta
            item["metadata"]["CarrierFrequencyOffset"] = self.df

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                times = item["metadata"]["dt"] * np.arange(0, len(tensor[0]), 1).astype(
                    np.float32
                )
                x = self.rads + 2 * np.pi * self.df * times
                rot = torch.tensor(
                    [[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]]
                ).float()
                tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    times = item["metadata"]["dt"] * np.arange(
                        0, len(tensor[0]), 1
                    ).astype(np.float32)
                    x = self.rads + 2 * np.pi * self.df * times
                    rot = torch.tensor(
                        [[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]]
                    ).float()
                    tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                self.theta = self.rng_theta.sample().item()
                self.rads = np.pi * self.theta / (180)
                self.df = (
                    self.rng_df.sample().item()
                )  # in Hz, if data_rate is 1, in percent of data_rate
                item["metadata"]["CarrierPhaseOffset"][idx] = self.theta
                item["metadata"]["CarrierFrequencyOffset"][idx] = self.df

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    times = item["metadata"]["dt"][idx].item() * np.arange(
                        0, len(tensor[0]), 1
                    ).astype(np.float32)
                    x = self.rads + 2 * np.pi * self.df * times
                    rot = torch.tensor(
                        [[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]]
                    ).float()
                    tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        times = item["metadata"]["dt"][idx].item() * np.arange(
                            0, len(tensor[0]), 1
                        ).astype(np.float32)
                        x = self.rads + 2 * np.pi * self.df * times
                        rot = torch.tensor(
                            [[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]]
                        ).float()
                        tensor_prime = torch.einsum("ijk,ik->jk", rot, tensor)
                        item["data"][idx] = tensor_prime

        return item


class AWGN(object):  # Noise transform, in dB.
    def __init__(self, SNRdB: int, batch=False, data_keys=False):
        self.SNRdB = int(SNRdB)
        # self.rng = np.random.default_rng() #this is not used any more
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            item["metadata"]["SNRdB"] = self.SNRdB

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                P = (tensor * tensor).sum(-1) / tensor.shape[
                    -1
                ]  # Actual power in the vector
                N0 = P / gamma  # Noise Power
                dist = torch.distributions.normal.Normal(
                    0, np.sqrt(N0 / 2.0)
                )  # actually two distributions
                n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                item["data"] = tensor + n  # received signal

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                    P = (tensor * tensor).sum(-1) / tensor.shape[
                        -1
                    ]  # Actual power in the vector
                    N0 = P / gamma  # Noise Power
                    dist = torch.distributions.normal.Normal(
                        0, np.sqrt(N0 / 2.0)
                    )  # actually two distributions
                    n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                    item[key] = tensor + n  # received signal

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                item["metadata"]["SNRdB"][idx] = self.SNRdB

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                    P = (tensor * tensor).sum(-1) / tensor.shape[
                        -1
                    ]  # Actual power in the vector
                    N0 = P / gamma  # Noise Power
                    dist = torch.distributions.normal.Normal(
                        0, np.sqrt(N0 / 2.0)
                    )  # actually two distributions
                    n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                    item["data"][idx] = tensor + n  # received signal

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                        P = (tensor * tensor).sum(-1) / tensor.shape[
                            -1
                        ]  # Actual power in the vector
                        N0 = P / gamma  # Noise Power
                        dist = torch.distributions.normal.Normal(
                            0, np.sqrt(N0 / 2.0)
                        )  # actually two distributions
                        n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                        item[key][idx] = tensor + n  # received signal

        return item


class RandomAWGN(object):  # Noise transform, in dB.
    def __init__(self, dBLow: int, dBHigh: int, batch=False, data_keys=False):
        self.low = dBLow
        self.hi = dBHigh
        # self.rng = np.random.default_rng() #this is not used any more
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            self.SNRdB = torch.randint(low=self.low, high=self.hi + 1, size=(1,)).item()
            item["metadata"]["SNRdB"] = self.SNRdB

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                P = (tensor * tensor).sum(-1) / tensor.shape[
                    -1
                ]  # Actual power in the vector
                N0 = P / gamma  # Noise Power
                dist = torch.distributions.normal.Normal(
                    0, np.sqrt(N0 / 2.0)
                )  # actually two distributions
                n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                item["data"] = tensor + n  # received signal

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                    P = (tensor * tensor).sum(-1) / tensor.shape[
                        -1
                    ]  # Actual power in the vector
                    N0 = P / gamma  # Noise Power
                    dist = torch.distributions.normal.Normal(
                        0, np.sqrt(N0 / 2.0)
                    )  # actually two distributions
                    n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                    item[key] = tensor + n  # received signal

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                self.SNRdB = torch.randint(
                    low=self.low, high=self.hi + 1, size=(1,)
                ).item()
                item["metadata"]["SNRdB"][idx] = self.SNRdB

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                    P = (tensor * tensor).sum(-1) / tensor.shape[
                        -1
                    ]  # Actual power in the vector
                    N0 = P / gamma  # Noise Power
                    dist = torch.distributions.normal.Normal(
                        0, np.sqrt(N0 / 2.0)
                    )  # actually two distributions
                    n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                    item["data"][idx] = tensor + n  # received signal

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        gamma = 10 ** (self.SNRdB / 10.0)  # SNR to linear scale
                        P = (tensor * tensor).sum(-1) / tensor.shape[
                            -1
                        ]  # Actual power in the vector
                        N0 = P / gamma  # Noise Power
                        dist = torch.distributions.normal.Normal(
                            0, np.sqrt(N0 / 2.0)
                        )  # actually two distributions
                        n = dist.sample(sample_shape=(tensor.shape[-1],)).T
                        item[key][idx] = tensor + n  # received signal

        return item


class Normalize_Amplitude(object):
    def __init__(self, amp=1, batch=False, data_keys=False):
        self.stabilizer = 1e-18
        self.amp = amp
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = (
                        self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                    )
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = (
                        self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                    )
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = (
                            self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                        )
                        item[key][idx] = tensor_prime

        return item


class Normalize_Amplitude_Range(object):
    def __init__(
        self,
        amp_mean=2.0,
        amp_std=0.5,
        amp_clip_low=1,
        amp_clip_hi=3,
        batch=False,
        data_keys=False,
    ):
        self.stabilizer = 1e-18
        # self.rng = np.random.default_rng() #no longer used

        self.amp_mean = amp_mean
        self.amp_std = amp_std
        self.amp_clip_low = amp_clip_low
        self.amp_clip_hi = amp_clip_hi

        self.rng = torch.distributions.normal.Normal(self.amp_mean, self.amp_std)

        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            amp = self.rng.sample().item()
            self.amp = np.clip(amp, self.amp_clip_low, self.amp_clip_hi)

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = (
                        self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                    )
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                amp = self.rng.sample().item()
                self.amp = np.clip(amp, self.amp_clip_low, self.amp_clip_hi)

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = (
                        self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                    )
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = (
                            self.amp * tensor / (abs(tensor).max() + self.stabilizer)
                        )
                        item[key][idx] = tensor_prime

        return item


class Normalize_Power(object):
    def __init__(self, batch=False, data_keys=False):
        self.stabilizer = 1e-18
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                P = (tensor * tensor).sum(-1) / tensor.shape[
                    -1
                ]  # Actual power in the vector
                tensor_prime = tensor / (P.sum() + self.stabilizer)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    P = (tensor * tensor).sum(-1) / tensor.shape[
                        -1
                    ]  # Actual power in the vector
                    tensor_prime = tensor / (P.sum() + self.stabilizer)
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    P = (tensor * tensor).sum(-1) / tensor.shape[
                        -1
                    ]  # Actual power in the vector
                    tensor_prime = tensor / (P.sum() + self.stabilizer)
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        P = (tensor * tensor).sum(-1) / tensor.shape[
                            -1
                        ]  # Actual power in the vector
                        tensor_prime = tensor / (P.sum() + self.stabilizer)
                        item[key][idx] = tensor_prime

        return item


class Random_Fading(object):
    def __init__(self, low, high, batch=False, data_keys=False):
        # dimensionless parameters that describe range for strength of NLOS multipath environment
        self.low = low
        self.high = high
        self.num_sin = 20  # number of sinusoids to sum
        # self.rng = np.random.default_rng() #no longer using this
        self.batch = batch
        self.keys = data_keys
        self.rng_str = torch.distributions.uniform.Uniform(self.low, self.high)
        self.rng_ph = torch.distributions.uniform.Uniform(0, 2 * np.pi)

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            # This block calculates the stuff
            self.max_shift = (
                self.rng_str.sample().item()
            )  # dimensionless parameter that describes strength
            # of NLOS multipath fading environment.
            item["metadata"]["fading"] = self.max_shift
            nlos_aoa = torch.cos(
                2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
            )  # angle of arrival for each sinusoid
            nlos_phase_a = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for in-phase
            nlos_phase_b = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for quadrature

            x = 0
            y = 0
            num_samp = item["data"].shape[-1]
            t = torch.linspace(0, 1, num_samp)  # time vector

            for i in range(self.num_sin):
                x += np.cos(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                )
                y += np.sin(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                )

            # normalization
            x = (1 / np.sqrt(self.num_sin)) * x
            y = (1 / np.sqrt(self.num_sin)) * y

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = torch.zeros_like(tensor)
                tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                    tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                # This block calculates the stuff
                self.max_shift = (
                    self.rng_str.sample().item()
                )  # dimensionless parameter that describes strength
                # of NLOS multipath fading environment.
                item["metadata"]["fading"] = self.max_shift
                nlos_aoa = torch.cos(
                    2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
                )  # angle of arrival for each sinusoid
                nlos_phase_a = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for in-phase
                nlos_phase_b = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for quadrature

                x = 0
                y = 0
                num_samp = item["data"].shape[-1]
                t = torch.linspace(0, 1, num_samp)  # time vector

                for i in range(self.num_sin):
                    x += np.cos(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                    )
                    y += np.sin(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                    )

                # normalization
                x = (1 / np.sqrt(self.num_sin)) * x
                y = (1 / np.sqrt(self.num_sin)) * y

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                    tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = torch.zeros_like(tensor)
                        tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                        tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                        item[key][idx] = tensor_prime

        return item


class Random_Amplitude_Fading(object):
    def __init__(self, low, high, batch=False, data_keys=False):
        # dimensionless parameters that describe range for strength of NLOS multipath environment
        self.low = low
        self.high = high
        self.num_sin = 20  # number of sinusoids to sum
        # self.rng = np.random.default_rng() #no longer using this
        self.batch = batch
        self.keys = data_keys
        self.rng_str = torch.distributions.uniform.Uniform(self.low, self.high)
        self.rng_ph = torch.distributions.uniform.Uniform(0, 2 * np.pi)

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            # This block calculates the stuff
            self.max_shift = (
                self.rng_str.sample().item()
            )  # dimensionless parameter that describes strength
            # of NLOS multipath fading environment.
            item["metadata"]["fading"] = self.max_shift
            nlos_aoa = torch.cos(
                2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
            )  # angle of arrival for each sinusoid
            nlos_phase_a = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for in-phase
            nlos_phase_b = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for quadrature

            x = 0
            y = 0
            num_samp = item["data"].shape[-1]
            t = torch.linspace(0, 1, num_samp)  # time vector

            for i in range(self.num_sin):
                x += np.cos(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                )
                y += np.sin(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                )

            # normalization
            x = (1 / np.sqrt(self.num_sin)) * x
            y = (1 / np.sqrt(self.num_sin)) * y
            z = np.sqrt(x**2 + y**2)

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = torch.zeros_like(tensor)
                tensor_prime[0, :] = tensor[0, :] * z
                tensor_prime[1, :] = tensor[1, :] * z
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * z
                    tensor_prime[1, :] = tensor[1, :] * z
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                # This block calculates the stuff
                self.max_shift = (
                    self.rng_str.sample().item()
                )  # dimensionless parameter that describes strength
                # of NLOS multipath fading environment.
                item["metadata"]["fading"] = self.max_shift
                nlos_aoa = torch.cos(
                    2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
                )  # angle of arrival for each sinusoid
                nlos_phase_a = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for in-phase
                nlos_phase_b = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for quadrature

                x = 0
                y = 0
                num_samp = item["data"].shape[-1]
                t = torch.linspace(0, 1, num_samp)  # time vector

                for i in range(self.num_sin):
                    x += np.cos(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                    )
                    y += np.sin(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                    )

                # normalization
                x = (1 / np.sqrt(self.num_sin)) * x
                y = (1 / np.sqrt(self.num_sin)) * y
                z = np.sqrt(x**2 + y**2)

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * z
                    tensor_prime[1, :] = tensor[1, :] * z
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = torch.zeros_like(tensor)
                        tensor_prime[0, :] = tensor[0, :] * z
                        tensor_prime[1, :] = tensor[1, :] * z
                        item[key][idx] = tensor_prime

        return item


class Fading(object):
    def __init__(self, max_shift, batch=False, data_keys=False):
        self.max_shift = max_shift  # dimensionless parameter that describes strength
        # of NLOS multipath fading environment.
        self.num_sin = 20  # number of sinusoids to sum
        # self.rng = np.random.default_rng() #defunct
        self.batch = batch
        self.keys = data_keys
        self.rng_ph = torch.distributions.uniform.Uniform(0, 2 * np.pi)

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            # This block calculates the stuff
            item["metadata"]["fading"] = self.max_shift
            nlos_aoa = torch.cos(
                2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
            )  # angle of arrival for each sinusoid
            nlos_phase_a = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for in-phase
            nlos_phase_b = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for quadrature

            x = 0
            y = 0
            num_samp = item["data"].shape[-1]
            t = torch.linspace(0, 1, num_samp)  # time vector

            for i in range(self.num_sin):
                x += np.cos(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                )
                y += np.sin(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                )

            # normalization
            x = (1 / np.sqrt(self.num_sin)) * x
            y = (1 / np.sqrt(self.num_sin)) * y

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = torch.zeros_like(tensor)
                tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                    tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                # This block calculates the stuff
                self.max_shift = (
                    self.rng_str.sample().item()
                )  # dimensionless parameter that describes strength
                # of NLOS multipath fading environment.
                item["metadata"]["fading"] = self.max_shift
                nlos_aoa = torch.cos(
                    2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
                )  # angle of arrival for each sinusoid
                nlos_phase_a = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for in-phase
                nlos_phase_b = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for quadrature

                x = 0
                y = 0
                num_samp = item["data"].shape[-1]
                t = torch.linspace(0, 1, num_samp)  # time vector

                for i in range(self.num_sin):
                    x += np.cos(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                    )
                    y += np.sin(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                    )

                # normalization
                x = (1 / np.sqrt(self.num_sin)) * x
                y = (1 / np.sqrt(self.num_sin)) * y

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                    tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = torch.zeros_like(tensor)
                        tensor_prime[0, :] = tensor[0, :] * x - tensor[1, :] * y
                        tensor_prime[1, :] = tensor[1, :] * x + tensor[0, :] * y
                        item[key][idx] = tensor_prime

        return item


class Amplitude_Fading(object):
    def __init__(self, max_shift, batch=False, data_keys=False):
        self.max_shift = max_shift  # dimensionless parameter that describes strength
        # of NLOS multipath fading environment.
        self.num_sin = 20  # number of sinusoids to sum
        # self.rng = np.random.default_rng() #defunct
        self.batch = batch
        self.keys = data_keys
        self.rng_ph = torch.distributions.uniform.Uniform(0, 2 * np.pi)

    def __call__(self, item):

        if not self.batch:  # NO BATCH
            # This block calculates the stuff
            item["metadata"]["fading"] = self.max_shift
            nlos_aoa = torch.cos(
                2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
            )  # angle of arrival for each sinusoid
            nlos_phase_a = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for in-phase
            nlos_phase_b = self.rng_ph.sample(
                sample_shape=(self.num_sin,)
            )  # random phases for quadrature

            x = 0
            y = 0
            num_samp = item["data"].shape[-1]
            t = torch.linspace(0, 1, num_samp)  # time vector

            for i in range(self.num_sin):
                x += np.cos(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                )
                y += np.sin(
                    2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                )

            # normalization
            x = (1 / np.sqrt(self.num_sin)) * x
            y = (1 / np.sqrt(self.num_sin)) * y
            z = np.sqrt(x**2 + y**2)

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = torch.zeros_like(tensor)
                tensor_prime[0, :] = tensor[0, :] * z
                tensor_prime[1, :] = tensor[1, :] * z
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * z
                    tensor_prime[1, :] = tensor[1, :] * z
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):
                # This block calculates the stuff
                self.max_shift = (
                    self.rng_str.sample().item()
                )  # dimensionless parameter that describes strength
                # of NLOS multipath fading environment.
                item["metadata"]["fading"] = self.max_shift
                nlos_aoa = torch.cos(
                    2 * np.pi * torch.arange(0, self.num_sin, 1) / self.num_sin
                )  # angle of arrival for each sinusoid
                nlos_phase_a = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for in-phase
                nlos_phase_b = self.rng_ph.sample(
                    sample_shape=(self.num_sin,)
                )  # random phases for quadrature

                x = 0
                y = 0
                num_samp = item["data"].shape[-1]
                t = torch.linspace(0, 1, num_samp)  # time vector

                for i in range(self.num_sin):
                    x += np.cos(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_a[i]
                    )
                    y += np.sin(
                        2 * np.pi * self.max_shift * t * nlos_aoa[i] + nlos_phase_b[i]
                    )

                # normalization
                x = (1 / np.sqrt(self.num_sin)) * x
                y = (1 / np.sqrt(self.num_sin)) * y
                z = np.sqrt(x**2 + y**2)

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = torch.zeros_like(tensor)
                    tensor_prime[0, :] = tensor[0, :] * z
                    tensor_prime[1, :] = tensor[1, :] * z
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = torch.zeros_like(tensor)
                        tensor_prime[0, :] = tensor[0, :] * z
                        tensor_prime[1, :] = tensor[1, :] * z
                        item[key][idx] = tensor_prime

        return item


class ADC_Quantize(object):
    def __init__(self, n_bits=14, range_use=1.0, batch=False, data_keys=False):
        self.n_bits = n_bits
        self.range_use = range_use
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                lvls = 2**self.n_bits

                if self.range_use > 1:
                    excess = self.range_use
                    self.range_use = 1
                    max_amp = abs(tensor).max() / self.range_use
                    q = max_amp / lvls
                    tensor_prime = q * np.round(tensor / q)
                    max_amp_2 = max_amp / excess
                    tensor_prime = np.clip(tensor_prime, -max_amp_2, max_amp_2)
                else:
                    excess = 1
                    max_amp = abs(tensor).max() / self.range_use
                    q = max_amp / lvls
                    tensor_prime = q * np.round(tensor / q)

                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]

                    lvls = 2**self.n_bits

                    if self.range_use > 1:
                        excess = self.range_use
                        self.range_use = 1
                        max_amp = abs(tensor).max() / self.range_use
                        q = max_amp / lvls
                        tensor_prime = q * np.round(tensor / q)
                        max_amp_2 = max_amp / excess
                        tensor_prime = np.clip(tensor_prime, -max_amp_2, max_amp_2)
                    else:
                        excess = 1
                        max_amp = abs(tensor).max() / self.range_use
                        q = max_amp / lvls
                        tensor_prime = q * np.round(tensor / q)

                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]

                    lvls = 2**self.n_bits

                    if self.range_use > 1:
                        excess = self.range_use
                        self.range_use = 1
                        max_amp = abs(tensor).max() / self.range_use
                        q = max_amp / lvls
                        tensor_prime = q * np.round(tensor / q)
                        max_amp_2 = max_amp / excess
                        tensor_prime = np.clip(tensor_prime, -max_amp_2, max_amp_2)
                    else:
                        excess = 1
                        max_amp = abs(tensor).max() / self.range_use
                        q = max_amp / lvls
                        tensor_prime = q * np.round(tensor / q)

                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]

                        lvls = 2**self.n_bits

                        if self.range_use > 1:
                            excess = self.range_use
                            self.range_use = 1
                            max_amp = abs(tensor).max() / self.range_use
                            q = max_amp / lvls
                            tensor_prime = q * np.round(tensor / q)
                            max_amp_2 = max_amp / excess
                            tensor_prime = np.clip(tensor_prime, -max_amp_2, max_amp_2)
                        else:
                            excess = 1
                            max_amp = abs(tensor).max() / self.range_use
                            q = max_amp / lvls
                            tensor_prime = q * np.round(tensor / q)

                        item[key][idx] = tensor_prime

        return item


class Normalize_Absolute(object):
    def __init__(self, max_amp, new_amp=1, batch=False, data_keys=False):
        self.stabilizer = 1e-18
        self.new_amp = new_amp
        self.max_amp = max_amp
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = self.new_amp * tensor / (self.max_amp + self.stabilizer)
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = (
                        self.new_amp * tensor / (self.max_amp + self.stabilizer)
                    )
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = (
                        self.new_amp * tensor / (self.max_amp + self.stabilizer)
                    )
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = (
                            self.new_amp * tensor / (self.max_amp + self.stabilizer)
                        )
                        item[key][idx] = tensor_prime

        return item


class Fix_Dtype(object):
    def __init__(self, batch=False, data_keys=False):
        self.type = "float32"
        self.batch = batch
        self.keys = data_keys

    def __call__(self, item):

        if not self.batch:  # NO BATCH

            if not self.keys:  # NO BATCH, NO KEYS
                tensor = item["data"]
                tensor_prime = tensor.float()
                item["data"] = tensor_prime

            else:  # NO BATCH, WITH DATA KEYS
                for key in self.keys:
                    tensor = item[key]
                    tensor_prime = tensor.float()
                    item[key] = tensor_prime

        else:  # WITH BATCH
            for idx in range(len(item["metadata"]["dt"])):

                if not self.keys:  # WITH BATCH, NO KEYS
                    tensor = item["data"][idx]
                    tensor_prime = tensor.float()
                    item["data"][idx] = tensor_prime

                else:  # WITH BATCH, WITH DATA KEYS
                    for key in self.keys:
                        tensor = item[key][idx]
                        tensor_prime = tensor.float()
                        item[key][idx] = tensor_prime

        return item


class DopplerShift(object):
    """Apply Doppler shift by fractional resampling."""

    def __init__(
        self,
        shift_hz: float,
        sample_rate: float,
        batch: bool = False,
        data_keys: bool | list[str] = False,
    ) -> None:
        self.shift_hz = shift_hz
        self.sample_rate = sample_rate
        self.batch = batch
        self.keys = data_keys

    def _resample(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resample ``tensor`` using linear interpolation."""
        target_len = int(
            round(
                tensor.shape[-1] * (self.sample_rate + self.shift_hz) / self.sample_rate
            )
        )
        x = tensor.unsqueeze(0)  # add batch dim for interpolate
        resampled = torch.nn.functional.interpolate(
            x, size=target_len, mode="linear", align_corners=False
        ).squeeze(0)
        if resampled.shape[-1] > tensor.shape[-1]:
            resampled = resampled[..., : tensor.shape[-1]]
        elif resampled.shape[-1] < tensor.shape[-1]:
            pad = tensor.shape[-1] - resampled.shape[-1]
            resampled = torch.nn.functional.pad(resampled, (0, pad))
        return resampled

    def __call__(self, item):
        if not self.batch:
            item["metadata"]["DopplerShiftHz"] = self.shift_hz
            if not self.keys:
                item["data"] = self._resample(item["data"])
            else:
                for key in self.keys:
                    item[key] = self._resample(item[key])
        else:
            for idx in range(len(item["metadata"]["dt"])):
                item["metadata"]["DopplerShiftHz"][idx] = self.shift_hz
                if not self.keys:
                    item["data"][idx] = self._resample(item["data"][idx])
                else:
                    for key in self.keys:
                        item[key][idx] = self._resample(item[key][idx])
        return item


class TDLRayleigh(object):
    """Apply a simple time-delay line Rayleigh fading model."""

    def __init__(
        self,
        delays: list[float],
        avg_gains_dB: list[float],
        sample_rate: float,
        batch: bool = False,
        data_keys: bool | list[str] = False,
    ) -> None:
        if len(delays) != len(avg_gains_dB):
            raise ValueError("`delays` and `avg_gains_dB` must be same length")
        self.delays = torch.tensor(delays, dtype=torch.float32)
        self.avg_gains_dB = torch.tensor(avg_gains_dB, dtype=torch.float32)
        self.sample_rate = sample_rate
        self.batch = batch
        self.keys = data_keys

    def _generate_impulse(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample_delays = torch.round(self.delays * self.sample_rate).to(torch.long)
        max_delay = int(sample_delays.max().item())
        h_real = torch.zeros(max_delay + 1)
        h_imag = torch.zeros(max_delay + 1)
        for d, gain_db in zip(sample_delays, self.avg_gains_dB):
            power = 10 ** (gain_db / 10)
            std = (power**0.5) / 2**0.5
            h_real[d] += torch.randn(1).squeeze() * std
            h_imag[d] += torch.randn(1).squeeze() * std
        return h_real, h_imag

    def _apply(
        self, tensor: torch.Tensor, h_real: torch.Tensor, h_imag: torch.Tensor
    ) -> torch.Tensor:
        x_r = tensor[0].unsqueeze(0).unsqueeze(0)
        x_i = tensor[1].unsqueeze(0).unsqueeze(0)
        k_r = h_real.flip(0).unsqueeze(0).unsqueeze(0)
        k_i = h_imag.flip(0).unsqueeze(0).unsqueeze(0)
        pad = k_r.shape[-1] - 1
        out_r = torch.nn.functional.conv1d(
            x_r, k_r, padding=pad
        ) - torch.nn.functional.conv1d(x_i, k_i, padding=pad)
        out_i = torch.nn.functional.conv1d(
            x_i, k_r, padding=pad
        ) + torch.nn.functional.conv1d(x_r, k_i, padding=pad)
        out_r = out_r.squeeze(0).squeeze(0)[: tensor.shape[-1]]
        out_i = out_i.squeeze(0).squeeze(0)[: tensor.shape[-1]]
        return torch.stack([out_r, out_i], dim=0)

    def __call__(self, item):
        h_r, h_i = self._generate_impulse()
        if not self.batch:
            if not self.keys:
                item["data"] = self._apply(item["data"], h_r, h_i)
            else:
                for key in self.keys:
                    item[key] = self._apply(item[key], h_r, h_i)
        else:
            for idx in range(len(item["metadata"]["dt"])):
                if not self.keys:
                    item["data"][idx] = self._apply(item["data"][idx], h_r, h_i)
                else:
                    for key in self.keys:
                        item[key][idx] = self._apply(item[key][idx], h_r, h_i)
        return item
