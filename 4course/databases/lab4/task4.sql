insert into sal_copy
values (1008, 'Sidorovich', 'Chernobyl', 0.8),
       (1009, 'Saharov', 'Yantar', 0.9);

select * from sal_copy;
delete from sal_copy where snum = 1009;
select * from sal_copy;