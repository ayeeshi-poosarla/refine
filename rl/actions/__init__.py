from rl.actions.add_direction_of_change import AddDirectionOfChange
from rl.actions.add_missingness_indicator import AddMissingnessIndicator
from rl.actions.remove_lowest_variance_field import RemoveLowestVarianceField

ALL_ACTIONS = [
    RemoveLowestVarianceField(),
    AddDirectionOfChange(),
    AddMissingnessIndicator(),
]
