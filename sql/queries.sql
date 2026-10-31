
-- Q1: Total revenue per city (HAVING revenue > 10000)
SELECT c.city, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'completed'
GROUP BY c.city
HAVING revenue > 10000;

-- Q2: Top 5 product categories by order count (3‑table join)
SELECT p.category, COUNT(*) AS order_cnt
FROM orders o
JOIN products p ON o.product_id = p.product_id
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'completed'
GROUP BY p.category
ORDER BY order_cnt DESC
LIMIT 5;

-- Q3: Customers with >3 orders and total spend > 500 (subquery)
SELECT customer_id, total_spend, order_cnt
FROM (
    SELECT o.customer_id,
           SUM(o.order_amount) AS total_spend,
           COUNT(*) AS order_cnt
    FROM orders o
    WHERE o.order_status = 'completed'
    GROUP BY o.customer_id
) sub
WHERE order_cnt > 3 AND total_spend > 500;

-- Q4: Daily running total of completed orders (window function)
SELECT DATE(order_timestamp) AS day,
       SUM(order_amount) OVER (ORDER BY DATE(order_timestamp) ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM orders
WHERE order_status = 'completed'
ORDER BY day;

-- Q5: Customers with no completed orders (left join)
SELECT c.customer_id, c.city, o.order_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_status = 'completed'
WHERE o.order_id IS NULL;

-- Q6: CTE – customers acquired after 2025‑01‑01 with their first order amount
WITH first_orders AS (
    SELECT customer_id, MIN(order_timestamp) AS first_ts
    FROM orders
    GROUP BY customer_id
)
SELECT c.customer_id, c.city, o.order_amount
FROM customers c
JOIN first_orders fo ON c.customer_id = fo.customer_id
JOIN orders o ON o.customer_id = c.customer_id AND o.order_timestamp = fo.first_ts
WHERE fo.first_ts >= '2025-01-01';

-- Q7: Rank customers by total spend (window ranking)
SELECT customer_id, total_spend,
       RANK() OVER (ORDER BY total_spend DESC) AS spend_rank
FROM (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE order_status = 'completed'
    GROUP BY customer_id
) sub;
