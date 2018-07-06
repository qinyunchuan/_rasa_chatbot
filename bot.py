# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings

from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.events import SlotSet
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)

support_search = ["消费", "流量"]


def extract_item(item):
    """
    check if item supported, this func just for lack of train data.
    :param item: item in track, eg: "流量"、"查流量"
    :return:
    """
    if item is None:
        return None
    for name in support_search:
        if name in item:
            return name
    return None


class ActionSearchConsume(Action):
    def name(self):
        return 'action_search_consume'

    def run(self, dispatcher, tracker, domain):
        item = tracker.get_slot("item")
        item = extract_item(item)
        if item is None:
            dispatcher.utter_message("您好，我现在只会查话费和流量")
            dispatcher.utter_message("你可以这样问我：“帮我查话费”")
            return []

        time = tracker.get_slot("time")
        if time is None:
            dispatcher.utter_message("您想查询哪个月的消费？")
            return []
        # query database here using item and time as key. but you may normalize time format first.
        dispatcher.utter_message("好，请稍等")
        if item == "流量":
            dispatcher.utter_message("您好，您{}共使用{}二百八十兆，剩余三十兆。".format(time, item))
        else:
            dispatcher.utter_message("您好，您{}共消费二十八元。".format(time))
        return []


class MobilePolicy(KerasPolicy):
 def model_architecture(self, input_shape, output_shape):
        """Build a Keras model and return a compiled model."""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        from keras.models import Sequential
        from keras.layers import \
            Masking, LSTM, Dense, TimeDistributed, Activation

        # Build Model
        model = Sequential()

        # the shape of the y vector of the labels,
        # determines which output from rnn will be used
        # to calculate the loss
        if len(output_shape) == 1:
            # y is (num examples, num features) so
            # only the last output from the rnn is used to
            # calculate the loss
            model.add(Masking(mask_value=-1, input_shape=input_shape))
            model.add(LSTM(self.rnn_size))
            model.add(Dense(input_dim=self.rnn_size, units=output_shape[-1]))
        elif len(output_shape) == 2:
            # y is (num examples, max_dialogue_len, num features) so
            # all the outputs from the rnn are used to
            # calculate the loss, therefore a sequence is returned and
            # time distributed layer is used

            # the first value in input_shape is max dialogue_len,
            # it is set to None, to allow dynamic_rnn creation
            # during prediction
            model.add(Masking(mask_value=-1,
                              input_shape=(None, input_shape[1])))
            model.add(LSTM(self.rnn_size, return_sequences=True))
            model.add(TimeDistributed(Dense(units=output_shape[-1])))
        else:
            raise ValueError("Cannot construct the model because"
                             "length of output_shape = {} "
                             "should be 1 or 2."
                             "".format(len(output_shape)))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model

def train_dialogue(domain_file="mobile_domain.yml",
                   model_path="models/dialogue",
                   training_data_file="data/mobile_story.md"):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(max_history=3), MobilePolicy()])

    training_data = agent.load_data(training_data_file);
    agent.train(
        training_data,
        epochs=400,
        batch_size=100,
        augmentation_factor=50,
        validation_split=0.2
    )

    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.converters import load_data
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Trainer

    training_data = load_data("data/mobile_nlu_data.json")
    trainer = Trainer(RasaNLUConfig("mobile_nlu_model_config.json"))
    trainer.train(training_data)
    model_directory = trainer.persist("models/", project_name="ivr", fixed_model_name="demo")

    return model_directory


def run_ivrbot_online(input_channel=ConsoleInputChannel(),
                      interpreter=RasaNLUInterpreter("models/default/model_20180706-192633"),
                      domain_file="mobile_domain.yml",
                      training_data_file="data/mobile_story.md"):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), KerasPolicy()],
                  interpreter=interpreter)

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


def run(serve_forever=True):
    agent = Agent.load("models/dialogue",
                       interpreter=RasaNLUInterpreter("models/default/model_20180706-192633"))

    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser(
        description="starts the bot")

    parser.add_argument(
        "task",
        choices=["train-nlu", "train-dialogue", "run", "online_train"],
        help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-dialogue":
        train_dialogue()
    elif task == "run":
        run()
    elif task == "online_train":
        run_ivrbot_online()
    else:
        warnings.warn("Need to pass either 'train-nlu', 'train-dialogue' or "
                      "'run' to use the script.")
        exit(1)
