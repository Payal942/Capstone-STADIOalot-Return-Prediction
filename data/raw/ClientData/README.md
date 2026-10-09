# STADIOalot — Order Return Prediction: Data Extract

Eight tables covering orders placed between January 2024 and June 2026. Timestamps are `yyyy-MM-dd HH:mm:ss` throughout. All monetary values are ZAR. `customer_id` and `seller_id` are pseudonymous and link across datasets.

## Files

### `table1_orders_transactions.csv`
One row per order-item. PREDICTOR table.  
Rows: 11,841

| Column | Description |
|---|---|
| `order_id` | String |
| `customer_id` | String |
| `order_datetime` | Datetime |
| `sku` | String |
| `seller_id` | String |
| `quantity` | Integer |
| `unit_price` | Numeric |
| `promotion_amount` | Numeric |
| `payment_method` | Categorical |
| `delivery_option` | Categorical |
| `customer_province` | Categorical |

### `table2_returns.csv`
One row per return event. TARGET CONSTRUCTION / evaluation only.  
Rows: 2,037

| Column | Description |
|---|---|
| `return_id` | String |
| `order_id` | String |
| `return_datetime` | Datetime |
| `return_receipt_datetime` | Datetime |
| `returned_quantity` | Integer |
| `return_reason` | Categorical |
| `return_condition` | Categorical |
| `refund_amount` | Numeric |
| `refund_outcome` | Categorical |
| `return_cost` | Numeric |
| `avoidable_return_flag` | Integer |

### `table3_product_catalogue.csv`
Product attributes as at the prediction point, keyed by order-item.  
Rows: 11,818

| Column | Description |
|---|---|
| `order_id` | String |
| `sku` | String |
| `product_category` | Categorical |
| `subcategory` | String |
| `brand` | Categorical |
| `size` | Categorical |
| `colour` | Categorical |
| `product_price` | Numeric |
| `product_rating` | Numeric |
| `review_count` | Integer |
| `listing_age_days` | Integer |

### `table4_marketplace_sellers.csv`
Seller measures as at the prediction point, keyed by order.  
Rows: 8,241

| Column | Description |
|---|---|
| `order_id` | String |
| `seller_id` | String |
| `seller_rating` | Numeric |
| `seller_tenure_months` | Integer |
| `average_shipping_days` | Numeric |
| `late_shipment_rate` | Numeric |
| `complaint_count` | Integer |

### `table5_customer_information.csv`
Customer attributes and spend history as at the prediction point.  
Rows: 8,241

| Column | Description |
|---|---|
| `order_id` | String |
| `customer_id` | String |
| `customer_age` | Integer |
| `customer_gender` | Categorical |
| `customer_province` | Categorical |
| `account_creation_date` | Datetime |
| `customer_tenure_months` | Integer |
| `preferred_shopping_channel` | Categorical |
| `average_order_value` | Numeric |
| `total_historical_spend` | Numeric |
| `previous_returned_orders` | Integer |

### `table6_customer_behaviour.csv`
30-day look-back behavioural counts ending at the prediction point.  
Rows: 8,241

| Column | Description |
|---|---|
| `order_id` | String |
| `customer_id` | String |
| `look_back_period_days` | Integer |
| `sessions` | Integer |
| `product_views` | Integer |
| `add_to_basket_count` | Integer |
| `previous_orders` | Integer |
| `previous_returns` | Integer |
| `customer_return_rate` | Numeric |

### `table7_inventory_warehouse.csv`
Stock and fulfilment position as at the prediction point, keyed by order-item.  
Rows: 11,818

| Column | Description |
|---|---|
| `order_id` | String |
| `sku` | String |
| `distribution_centre` | Categorical |
| `stock_on_hand` | Integer |
| `stock_age_days` | Integer |
| `historical_pick_pack_time` | Numeric |
| `inbound_datetime` | Datetime |

### `table8_delivery_logistics.csv`
Delivery estimates (pre-purchase) and actuals (post-purchase).  
Rows: 8,241

| Column | Description |
|---|---|
| `order_id` | String |
| `estimated_delivery_datetime` | Datetime |
| `delivery_option` | Categorical |
| `delivery_province` | Categorical |
| `route_distance_km` | Numeric |
| `historical_delivery_time_days` | Numeric |
| `historical_delivery_failure_rate` | Numeric |
| `actual_delivery_datetime` | Datetime |
| `delivery_attempts` | Integer |
| `on_time_flag` | Integer |

## Data quality notes

This is an extract from live operational systems and has **not** been cleaned. Expect:

- Missing values in optional / conditionally-populated fields
- Inconsistent casing and stray whitespace in some free-text and categorical fields
- A small number of duplicate records
- Occasional out-of-range or implausible values from data-capture errors
- Some date fields captured in inconsistent formats

**Agreed target definition.** The unit of prediction is the **order**. An order is `Return Status = 1 (Returned)` if **at least one item** from it is returned within a **30-day return observation window** from the order date; otherwise 0 (Not Returned). Build the label by left-joining `table2_returns.csv` to the distinct `order_id` list in `table1_orders_transactions.csv` — orders with no return row are 0. Every return in this extract falls inside the 30-day window, so no additional filtering is needed.

**Point-in-time guarantee — this was the core of the request.** Tables 3-8 are keyed on `order_id` (and `sku` where item-level) rather than on the entity, precisely because their values change over time. Every measure in them is calculated using **only information available at or before that order's timestamp**:
- `product_rating`, `review_count`, `listing_age_days` — as at the order, not today.
- `seller_rating`, `seller_tenure_months`, `average_shipping_days`, `late_shipment_rate`, `complaint_count` — from prior orders only.
- `customer_tenure_months`, `average_order_value`, `total_historical_spend`, `previous_returned_orders`, `previous_orders`, `previous_returns`, `customer_return_rate` — exclude the current order.
- `stock_on_hand`, `stock_age_days`, `historical_pick_pack_time`, `inbound_datetime` — the position at the prediction point.
- `historical_delivery_time_days`, `historical_delivery_failure_rate` — from prior deliveries.

**Fields that are NOT predictive features** (post-purchase, supplied for target construction, evaluation or business-impact analysis only):
- Everything in `table2_returns.csv`.
- In `table8_delivery_logistics.csv`: `actual_delivery_datetime`, `delivery_attempts`, `on_time_flag`. The `estimated_delivery_datetime`, `delivery_option`, `delivery_province`, `route_distance_km` and the two historical rates **are** available at purchase and can be used.

**`avoidable_return_flag`:** STADIOalot does maintain this classification. A return is flagged avoidable where the reason is one STADIOalot could have prevented — sizing information, listing accuracy, packaging or picking error — as opposed to a genuine change of mind. It is known only after the return and is not a predictive feature.

**Scales, as requested:** `product_rating` and `seller_rating` are both **0-5**, one decimal. `late_shipment_rate`, `historical_delivery_failure_rate` and `customer_return_rate` are proportions 0-1. `historical_pick_pack_time` is minutes; `historical_delivery_time_days` is days.

**Note on `average_order_value`:** blank for a customer's first order, since there are no eligible prior orders to average.

_Synthetic data supplied for the CAP182 capstone project. Not real client data._