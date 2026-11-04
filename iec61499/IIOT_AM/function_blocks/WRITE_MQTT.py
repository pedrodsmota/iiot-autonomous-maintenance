

class WRITE_MQTT:

    def schedule(self, event_name, event_value, topic, message, client):
        init_o = None
        run_o = None

        if event_name == 'INIT':
            init_o = 1

        # publishes the value
        elif event_name == 'RUN':
            client.publish(topic, str(message))
            run_o = 1

        return [init_o, run_o]
