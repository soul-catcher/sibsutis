select cust.cname, sal.sname from cust, sal where cust.city = sal.city
and (cust.city = '&1' OR sal.city = '&1');