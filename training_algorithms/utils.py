class LearningRateAdjuster(object):
    def __init__(self, shrink_time, init_value, dc_rate):
        self.shrink_time = shrink_time
        self.init_value = init_value
        self.dc_rate = dc_rate

    def get_value(self, epoch):
        if epoch > self.shrink_time:
            return self.init_value / (1. + self.dc_rate * epoch)
        else:
            return self.init_value

class MomentumAdjuster(object):
    def __init__(self, inc_start, inc_end, init_value, final_value):
        self.inc_start = inc_start
        self.inc_end = inc_end
        self.init_value = init_value
        self.final_value = final_value

    def get_value(self, epoch):
        if epoch < self.inc_start:
            return self.init_value
        elif epoch < self.inc_end:
            return self.init_value + ((self.final_value - self.init_value) / \
                    (self.inc_end - self.inc_start)) * (epoch - self.inc_start)
        else:
            return self.final_value


