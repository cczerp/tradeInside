# sites.py
SITES = {
    "quiver": {
        "insider_trading": "https://www.quiverquant.com/insiders/",
        "institutional_13f": "https://www.quiverquant.com/sec13f/",
        "lobbying": "https://www.quiverquant.com/lobbying/",
    },
    "openinsider": {
        "latest_purchases": "http://openinsider.com/insider-purchases",  # All buys - 100 rows
        "latest_sales": "http://openinsider.com/insider-sales",  # All sells - 100 rows
        "penny_buys": "http://openinsider.com/latest-penny-stock-buys",  # Penny stocks - 100 rows
        "cluster_buys": "http://openinsider.com/latest-cluster-buys",  # Multiple insiders - 100 rows
        "ceo_buys_25k": "http://openinsider.com/latest-ceo-cfo-purchases-25k",  # CEO/CFO buys >$25k
        "ceo_sales_25k": "http://openinsider.com/latest-ceo-cfo-sales-25k",  # CEO/CFO sells >$25k
        "ceo_buys_100k": "http://openinsider.com/latest-ceo-cfo-purchases-100k",  # CEO/CFO buys >$100k
        "ceo_sales_100k": "http://openinsider.com/latest-ceo-cfo-sales-100k",  # CEO/CFO sells >$100k
        "top_officer_week": "http://openinsider.com/top-officer-purchases-of-the-week",  # Weekly top
        "top_officer_month": "http://openinsider.com/top-officer-purchases-of-the-month", 
    }
}
    