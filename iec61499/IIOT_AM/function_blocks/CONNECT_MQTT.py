import paho.mqtt.client as mqtt
from threading import Event


class CONNECT_MQTT:

    def __init__(self):
        self.client = None
        self.messages = None
        self.new_message = Event()
        self.new_message.clear()

        self.address = None
        self.port = None
        self.timeout = None

    def on_messages(self, client, userdata, message):
        # print("message received ", str(message.payload.decode("utf-8")))
        # print("message topic=", message.topic)

        self.messages[message.topic] = str(message.payload.decode("utf-8"))
        self.new_message.set()

    def connect(self):
        self.client = mqtt.Client()
        self.client.connect(self.address, self.port, self.timeout)
        self.client.on_message = self.on_messages
        self.client.loop_start()
        self.messages = dict()

    def schedule(self, event_name, event_value, address, port, timeout):
        init_event = None
        read_event = None
        return_messages = None

        if event_name == 'INIT':
            self.address = str(address)
            self.port = int(port)
            self.timeout = int(timeout)
            # connect to the broker
            self.connect()
            # update the event
            init_event = 1

        elif event_name == 'READ':
            # wait for new message
            self.new_message.wait()
            self.new_message.clear()
            return_messages = self.messages
            self.messages = dict()
            # update the event
            read_event = 1

        return [init_event, read_event, self.client, return_messages]

    def __del__(self):
        self.client.loop_stop()
        self.client.disconnect()
