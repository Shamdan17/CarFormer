from collections import OrderedDict
from dictor import dictor

import copy

from srunner.scenarioconfigs.route_scenario_configuration import (
    RouteScenarioConfiguration,
)


from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.checkpoint_tools import (
    fetch_dict,
    create_default_json_msg,
    save_dict,
)


def progress_less_than_threshold(record, threshold=70):
    return record["scores"]["score_route"] >= threshold


class RouteIndexer:
    def __init__(self, routes_file, scenarios_file, repetitions):
        self._routes_file = routes_file
        self._scenarios_file = scenarios_file
        self._repetitions = repetitions
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.routes_length = []
        self._index = 0

        # retrieve routes
        route_configurations = RouteParser.parse_routes_file(
            self._routes_file, self._scenarios_file, False
        )

        self.n_routes = len(route_configurations)
        self.total = self.n_routes * self._repetitions

        for i, config in enumerate(route_configurations):
            for repetition in range(repetitions):
                config.index = i * self._repetitions + repetition
                config.repetition_index = repetition
                self._configs_dict["{}.{}".format(config.name, repetition)] = copy.copy(
                    config
                )

        self._configs_list = list(self._configs_dict.items())

    def peek(self):
        return not (self._index >= len(self._configs_list))

    def next(self):
        if self._index >= len(self._configs_list):
            return None

        if hasattr(self, "repeat_indices"):
            while self._index not in self.repeat_indices:
                self._index += 1
                if self._index >= len(self._configs_list):
                    return None

        key, config = self._configs_list[self._index]
        self._index += 1

        return config

    def resume(self, endpoint, done_fnc=progress_less_than_threshold):
        data = fetch_dict(endpoint)

        if data:
            checkpoint_dict = dictor(data, "_checkpoint")
            if checkpoint_dict and done_fnc is not None and False:
                assert "records" in checkpoint_dict
                # assert (
                #     checkpoint_dict["progress"][0] == checkpoint_dict["progress"][1]
                # ), "Cannot resume before first pass is complete. "
                # assert checkpoint_dict["progress"][1] == len(
                #     checkpoint_dict["records"]
                # ), "Cannot resume before first pass is complete. length found: {} length expected: {}".format(
                #     len(checkpoint_dict["records"]), checkpoint_dict["progress"][1]
                # )
                # assert self.total == len(checkpoint_dict["records"]), (
                #     "Number of routes in checkpoint does not match number of routes in "
                #     "current run. "
                # )

                repeat_indices = []
                for i, record in enumerate(checkpoint_dict["records"]):
                    if not done_fnc(record):
                        repeat_indices.append(i)
                self.repeat_indices = repeat_indices
                self._index = repeat_indices[0] if repeat_indices else self.total
                print("Indices to repeat: {}".format(repeat_indices))
                print("Starting from index {}".format(self._index))

            elif checkpoint_dict and "progress" in checkpoint_dict:
                progress = checkpoint_dict["progress"]
                if not progress:
                    current_route = 0
                else:
                    current_route, total_routes = progress
                if current_route <= self.total:
                    self._index = current_route
                else:
                    print(
                        "Problem reading checkpoint. Route id {} "
                        "larger than maximum number of routes {}".format(
                            current_route, self.total
                        )
                    )

    def save_state(self, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()
        data["_checkpoint"]["progress"] = [self._index, self.total]

        save_dict(endpoint, data)
