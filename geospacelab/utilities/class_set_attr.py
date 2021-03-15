
    def set_attr(self, append=False, **kwargs):
        append_rec = 0
        for key, value in kwargs.items():
            if not hasattr(self, key):
                if not append:
                    StreamLogger.warning("%s is not found in the named attributes!", key)
                    append_rec = 1
                    continue
            if key == 'visual':
                self.visual.set_attr(**value)
            else:
                setattr(self, key, value)
        if append_rec:
            StreamLogger.info("To add the new attribute, use the keyword append, e.g., append=True")