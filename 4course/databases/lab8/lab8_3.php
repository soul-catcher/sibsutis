<?php
	$right_answers = array(6, 9, 4, 1, 3, 2, 5, 8, 7);
	if (!empty($_POST['name'])) {
		$name = htmlspecialchars($_POST['name']);
		$str = "$name, Вы";
	} else {
		$str = "Username, Вы";
	}
	if (!empty($_POST['answers'])) {
		$answers = $_POST['answers'];
		$count = 0;
		for ($i=0; $i < count($answers); $i++) { 
			if ($answers[$i] == $right_answers[$i]) {
				$count++;
			}
		}
		switch ($count) {
			case 9:
				print "$str великолепно знаете географию";
				break;

			case 8:
				print "$str отлично знаете географию";
				break;

			case 7:
				print "$str очень хорошо знаете географию";
				break;

			case 6:
				print "$str хорошо знаете географию";
				break;

			case 5:
				print "$str удовлетворительно знаете географию";
				break;

			case 4:
				print "$str терпимо знаете географию";
				break;

			case 3:
				print "$str плохо знаете географию";
				break;

			case 2:
				print "$str очень плохо знаете географию";
				break;
			
			default:
				print "$str вообще не знаете географию";
				break;
		}
	}
?>