
class READ_MQTT:

    def schedule(self, event_name, event_value, sub_topic, client, messages):
        init_o = None
        run_o = None
        subs_o = None
        message_str = ''

        if event_name == 'INIT':
            init_o = 1

        elif event_name == 'SUBS_TOPIC':
            # subscribe to the topic
            client.subscribe(sub_topic)

        # selects the message
        elif event_name == 'RUN':
            # parses the message
            if sub_topic in messages:
                message_str = messages[sub_topic]
                run_o = 1

        return [init_o, subs_o, run_o, message_str]
