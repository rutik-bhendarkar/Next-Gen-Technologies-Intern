# Week 1 Testing Record

## Automated test command

```bash
python -m unittest discover -s tests -v
```

## Test areas

- Total marks, percentage and grade calculation
- Grade boundary checks
- JSON save/load persistence
- Student search by ID and name
- Student update
- Student deletion
- Duplicate student ID validation

## Manual test checklist

- [x] Add student with valid data
- [x] View saved records
- [x] Search by student ID
- [x] Search by student name
- [x] Update student details
- [x] Delete a student after confirmation
- [x] Calculate percentage and grade
- [x] Check highest and lowest performer
- [x] Enter invalid numeric input
- [x] Verify records remain after restarting the program

## Clean end-to-end test

A separate clean project copy was used to test the full workflow: add -> search -> update -> performance summary -> delete -> view -> exit. All checks passed and no traceback occurred. See `END_TO_END_TEST.txt`.
