SELECT
questions.Id as QuestionId,
Title,
Tags,
Body,
Answers.AnswerBody,
AcceptedAnswerId,
questions.Score as QuestionScore,
ViewCount,
AnswerCount,
CommentCount,
FavoriteCount,
questions.CreationDate as QuestionCreationDate,
LastEditDate,
LastActivityDate,
ClosedDate
FROM Posts as questions
LEFT OUTER JOIN ( SELECT Id as AnswerId, Body as AnswerBody
FROM Posts WHERE PostTypeId = 2) as Answers
ON questions.AcceptedAnswerId = answers.AnswerId
WHERE questions.PostTypeId = 1
AND questions.CreationDate >= '01-01-2019 00:00:00'
AND questions.CreationDate < '01-01-2021 00:00:00'
AND (questions.Title LIKE '%Clion%'
OR questions.Title LIKE '%CLion%')
AND questions.score > 0
AND questions.DeletionDate IS NULL
AND ParentId IS NULL
AND questions.CommunityOwnedDate IS NULL