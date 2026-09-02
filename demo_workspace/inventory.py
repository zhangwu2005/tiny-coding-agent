"""Inventory reservation example with intentional bugs."""


def reserve_stock(stock, requests):
    """Reserve requested quantities and return remaining stock."""
    for sku, quantity in requests:
        if sku not in stock or stock[sku] < quantity:
            raise ValueError("insufficient stock")
        stock[sku] -= quantity
    return stock
