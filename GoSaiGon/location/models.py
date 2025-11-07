from datetime import time

from django.contrib.auth import get_user_model
from django.db import models
from django.db.models import Q, F

User = get_user_model()

## Custom QuerySet
class LocationQuerySet(models.QuerySet):
    def open_at(self, desired_time):
        return self.filter(
            Q(open_time__lte=desired_time, close_time__gte=desired_time)
            | Q(open_time__gt=F("close_time")) & (
                    Q(open_time__lte=desired_time) | Q(close_time__gte=desired_time))
        )


## Location List
class Location_List(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="location_lists",
    )
    name = models.CharField(max_length=50, default="")

    class Meta:
        ordering = ("name",)
        verbose_name = "Location List"
        verbose_name_plural = "Location Lists"

    def __str__(self):
        return self.name


## Location
class Location(models.Model):
    favourited_by = models.ManyToManyField(
        User,
        related_name="favourite_locations",
        blank=True
    )
    code = models.CharField(max_length=10, unique=True)
    location = models.CharField(max_length=64)
    type = models.CharField(max_length=18, default="")
    tags = models.TextField(default="")
    rating = models.FloatField(default=5.0)
    open_time = models.TimeField(default=time(0, 0))
    close_time = models.TimeField(default=time(23, 59))
    ticket_info = models.CharField(max_length=100, default="")
    address = models.CharField(max_length=100, default="")
    image_path = models.CharField(max_length=255, default="")
    description = models.TextField(default="")
    long_description = models.TextField(default="")
    coordinate = models.CharField(max_length=40, default="")

    objects = LocationQuerySet.as_manager()

    class Meta:
        verbose_name = "Location"
        verbose_name_plural = "Locations"
        ordering = ("location",)

    def __str__(self):
        return f"{self.code} - {self.location}"


## Comment
class Comment(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="comments"
    )
    location = models.ForeignKey(
        Location,
        on_delete=models.CASCADE,
        related_name="comments"
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="replies"
    )
    content = models.TextField()
    rating = models.IntegerField(null=True, blank=True)
    bot_reply = models.TextField(blank=True)
    is_edited = models.BooleanField(default=False)
    is_flagged = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "Comment"
        verbose_name_plural = "Comments"

    def __str__(self):
        return f"Comment by {self.user.username} on {self.location.location}"
