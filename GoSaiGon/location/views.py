from datetime import datetime

import joblib
import os
import spacy
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from location.models import Location, Comment

# ============================================================================
# INITIALIZATION
# ============================================================================

nlp = spacy.load("en_core_web_sm")

pipeline_path = os.path.join(settings.BASE_DIR, 'location', 'svm_tfidf_pipeline.pkl')
label_encoder_path = os.path.join(settings.BASE_DIR, 'location', 'label_encoder.pkl')

pipeline = joblib.load(pipeline_path)
label_encoder = joblib.load(label_encoder_path)

DEFAULT_FEEDBACK_REPLY = "Thanks for your feedback!"

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

def predict_sentiment(text):
    """Predict sentiment of text using trained SVM model."""
    if not text or not isinstance(text, str):
        return "Invalid input"

    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    cleaned_text = ' '.join(tokens)

    if not cleaned_text:
        return "Text too short or meaningless"

    pred_label = pipeline.predict([cleaned_text])[0]
    sentiment = label_encoder.inverse_transform([pred_label])[0]
    return sentiment


def get_bot_reply_from_rating(rating):
    """Generate bot reply based on rating."""
    replies = {
        5: "Awesome! We're thrilled you loved it!",
        4: "Great! Glad you had a good time.",
        3: "Thanks! We'll try to make your next visit even better.",
        2: "Sorry to hear that. We hope things improve.",
        1: "We sincerely apologize. Your feedback is valuable to us."
    }
    return replies.get(rating, DEFAULT_FEEDBACK_REPLY)


def get_bot_reply_from_sentiment(sentiment):
    """Generate bot reply and rating from sentiment analysis."""
    sentiment_map = {
        "positive": ("We're thrilled you had a great time! Hope to see you again!", 4),
        "negative": ("We're sorry to hear that. Your feedback helps us get better.", 2),
    }
    return sentiment_map.get(
        sentiment,
        ("Thank you for sharing your thoughts. We appreciate your input!", 3)
    )


# ============================================================================
# STAR RATING UTILITIES
# ============================================================================

def generate_star_html(rating):
    """Generate HTML for star rating display."""
    rating = round(rating * 2) / 2 if rating else 0
    full_stars = int(rating)
    has_half = (rating - full_stars) >= 0.5

    star_html = '<i class="fas fa-star"></i>' * full_stars

    if has_half:
        star_html += '<i class="fas fa-star-half-alt"></i>'
        empty_stars = 5 - full_stars - 1
    else:
        empty_stars = 5 - full_stars

    star_html += '<i class="far fa-star"></i>' * empty_stars
    return star_html


def get_favourite_symbol(user, location):
    """Get heart icon based on favourite status."""
    if user and user.is_authenticated and location.favourited_by.filter(id=user.id).exists():
        return '<i class="fa-solid fa-heart"></i>'
    return '<i class="fa-regular fa-heart"></i>'


def format_opening_hours(open_time, close_time):
    """Format opening hours display string."""
    open_str = open_time.strftime("%H:%M") if open_time else "N/A"
    close_str = close_time.strftime("%H:%M") if close_time else "N/A"

    if open_str == "00:00" and close_str == "23:59":
        return "All day"
    if close_time and open_time and close_time < open_time:
        return f"{open_str} - {close_str} (The next day)"
    return f"{open_str} - {close_str}"


# ============================================================================
# VIEWS
# ============================================================================

def overall_homepage(request):
    """Display homepage with featured locations."""
    locations = Location.objects.all()[:6]

    processed_locations = [
        {
            'code': loc.code,
            'location': loc.location,
            'description': loc.description,
            'image_path': loc.image_path,
            'rating': loc.rating,
            'star_html': generate_star_html(loc.rating),
        }
        for loc in locations
    ]

    return render(request, "homepage.html", {
        "all_of_locations": processed_locations,
    })


def locations(request):
    """Display and filter location listing."""
    if request.method == "POST":
        return handle_favourite_toggle(request)

    return handle_location_listing(request)


def handle_favourite_toggle(request):
    """Handle POST request to toggle favourite status."""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'unauthenticated'}, status=401)

    code = request.POST.get('value')
    if not code:
        return redirect('favourite')

    try:
        location = Location.objects.get(code=code)
    except Location.DoesNotExist:
        return JsonResponse({'error': 'Location not found'}, status=404)

    if request.user in location.favourited_by.all():
        location.favourited_by.remove(request.user)
    else:
        location.favourited_by.add(request.user)

    return redirect('locations')


def handle_location_listing(request):
    """Handle GET request with filters for location listing."""
    type_filter = request.GET.get('type')
    min_rating = request.GET.get('rating')
    desired_time = request.GET.get('desired_time')
    search_query = request.GET.get('search')

    queryset = Location.objects.all()

    # Apply filters
    if type_filter:
        queryset = queryset.filter(type__iexact=type_filter)

    if min_rating:
        try:
            queryset = queryset.filter(rating__gte=float(min_rating))
        except ValueError:
            pass

    if desired_time:
        try:
            time_obj = datetime.strptime(desired_time, "%H:%M").time()
            queryset = queryset.filter(
                open_time__lte=time_obj,
                close_time__gte=time_obj
            )
        except ValueError:
            pass

    if search_query:
        queryset = queryset.filter(
            Q(location__icontains=search_query) |
            Q(address__icontains=search_query) |
            Q(tags__icontains=search_query)
        )

    queryset = queryset.order_by('open_time')

    processed_locations = [
        {
            'code': loc.code,
            'location': loc.location,
            'description': loc.description,
            'image_path': loc.image_path,
            'rating': loc.rating,
            'open_time': format_opening_hours(loc.open_time, loc.close_time),
            'star_html': generate_star_html(loc.rating),
            'favourite_symbol': get_favourite_symbol(request.user, loc),
        }
        for loc in queryset
    ]

    return render(request, "locations.html", {
        'locations': processed_locations,
        'current_filters': {
            'type': type_filter or '',
            'rating': min_rating or '',
            'desired_time': desired_time or '',
            'search': search_query or '',
        }
    })


def display_location(request, location_code):
    """Display detailed location page with comments."""
    location = get_object_or_404(Location, code=location_code)

    if request.method == 'POST':
        return handle_comment_submission(request, location)

    return handle_location_display(request, location)


def handle_comment_submission(request, location):
    """Handle comment form submission."""
    content = request.POST.get('content', '').strip()
    rating = request.POST.get('rating')

    if not content:
        return redirect('display_location', location_code=location.code)

    if rating:
        try:
            rating = int(rating)
            bot_reply = get_bot_reply_from_rating(rating)
        except ValueError:
            rating = 3
            bot_reply = DEFAULT_FEEDBACK_REPLY
    else:
        sentiment = predict_sentiment(content)
        bot_reply, rating = get_bot_reply_from_sentiment(sentiment)

    Comment.objects.create(
        location=location,
        user=request.user,
        content=content,
        rating=rating,
        bot_reply=bot_reply
    )

    return redirect('display_location', location_code=location.code)


def handle_location_display(request, location):
    """Prepare context for location detail page."""
    comments = Comment.objects.filter(
        location=location,
        parent=None
    ).prefetch_related('replies').order_by('-created_at')

    lat, long = location.coordinate.split(", ")

    context = {
        "code": location.code,
        "location_name": location.location,
        "type": location.type,
        "open_time": format_opening_hours(location.open_time, location.close_time),
        "ticket_info": location.ticket_info,
        "address": location.address,
        "image_path": location.image_path,
        "long_description": location.long_description,
        "favourite_symbol": get_favourite_symbol(request.user, location),
        "lat": lat,
        "long": long,
        "star_html": generate_star_html(location.rating),
        "comments": comments,
        "location_obj": location
    }

    return render(request, "display.html", context)


@require_POST
@login_required
def submit_comment_ajax(request, location_code):
    """Handle AJAX comment submission."""
    content = request.POST.get('content', '').strip()
    rating = request.POST.get('rating')
    location = get_object_or_404(Location, code=location_code)

    if not content:
        return JsonResponse({'error': 'Empty content'}, status=400)

    if rating:
        try:
            rating = int(rating)
            bot_reply = get_bot_reply_from_rating(rating)
        except ValueError:
            rating = 3
            bot_reply = DEFAULT_FEEDBACK_REPLY
    else:
        sentiment = predict_sentiment(content)
        bot_reply, rating = get_bot_reply_from_sentiment(sentiment)

    comment = Comment.objects.create(
        location=location,
        user=request.user,
        content=content,
        rating=rating,
        bot_reply=bot_reply
    )

    return JsonResponse({
        'username': request.user.username,
        'content': comment.content,
        'bot_reply': comment.bot_reply
    })
