drop table sal_copy;

create table sal_copy as
select *
from sal;

alter table sal_copy
    add constraint primary key (snum);

describe sal_copy;
describe sal;
