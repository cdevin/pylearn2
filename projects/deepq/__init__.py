


class Gater():


    def fprop(self, state_below):

        experts = [expert.fprop(state_below) for expert in self.experts]
        # concat here
        z = self.mlp.fprop(state_below) * experts

        return z
