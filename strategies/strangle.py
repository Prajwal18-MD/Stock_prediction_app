class StrangleStrategy:
    def __init__(self, call_price, put_price, call_strike, put_strike):
        self.call_price = call_price
        self.put_price = put_price
        self.call_strike = call_strike
        self.put_strike = put_strike

    def calculate_strangle(self):
        """
        Calculate the cost of the strangle strategy.
        """
        return self.call_price + self.put_price
