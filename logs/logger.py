from datetime import datetime

class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, level, tag, msg):
        dt = datetime.now()
        dt_str = dt.strftime('%m/%d/%y %H:%M:%S.%f')[:-3]
        msg = f'{dt_str} [{level}] {tag} : {msg}\n'
        with open(self.path, 'a') as file:
            file.write(msg)

    def debug(self, tag, msg):
        self.log("DEBUG", tag, msg)

    def info(self, tag, msg):
        self.log("INFO", tag, msg)

    def warning(self, tag, msg):
        self.log("WARNING", tag, msg)

    def error(self, tag, msg):
        self.log("ERROR", tag, msg)

    def critical(self, tag, msg):
        self.log("CRITICAL", tag, msg)