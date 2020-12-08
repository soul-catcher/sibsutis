<?php
	// if (!$_POST['record']) {
	// 	print "!POST";
	// } else {
		// print "POST";
		$record = $_POST['record'];
		// print_r($record);
		// $record = $_POST['record'];
		if ($record['name'] == "" || $record['email'] == "") {
		// if (!isset($record['name']) || !isset($record['email'])) {
			print "<html><head><meta charset=utf-8></head><body><p>Поля, помеченные [*], являются обязательными для заполнения!</p></body></html>";
		} else {
			$link = mysqli_connect("localhost", "root", "root", "sample");
			if (mysqli_connect_errno()) {
				printf("Не удалось подключиться: %s\n", mysqli_connect_error());
	    		exit();
	    	}
			// $name = $record['name'];
			// $city = $record['city'];
			// $address = $record['address'];
			// $dob = $record['dob'];
			// $email = $record['email'];
			// print $email;
			// $query = "INSERT INTO notebook_br06 (name, city, address, birthday, mail) VALUES (" .$record['name']. ", " .$record['city']. ", " .$record['address']. ", " .$record['dob']. ", " .$record['email'] .")";
			$sql = "INSERT INTO notebook_br06 (name, city, address, birthday, mail) VALUES ('{$record['name']}', '{$record['city']}', '{$record['address']}', '{$record['dob']}', '{$record['email']}')";
			// mysqli_query($link, $sql);
			// $sql1 = "SELECT id, name FROM notebook_br06";
			// $result = mysqli_query($link, $sql1);
			if (mysqli_query($link, $sql)) {
			  echo "Новая запись добавлена успешно";
			} else {
			  echo "Ошибка: " . $sql . "<br>" . mysqli_error($link);
			}
			// print gettype($name);
			// print "<p>Запись успешно добавлена</br></br>";
		// 	if (mysqli_num_rows($result) > 0) {
  // // output data of each row
		// 	while ($row = mysqli_fetch_assoc($result)) {
		// 	    echo "id: " . $row["id"]. " - Name: " . $row["name"]. " " ."<br>";
		// 	  	}
		// 	} else {
		// 	  echo "0 results";
		// 	}
			// $db_name[] = mysqli_query($link, "SELECT name FROM notebook_br06");
			// $i = 0;
			// while ($i < count($db_name)) {
			// 	print "$db_name</br>";
			// }
			mysqli_close($link);
		}
	//}
?>